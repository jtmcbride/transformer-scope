import { beforeAll, expect, it } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { RunEngine } from '../../src/engine/runs';
import { TransformerModel, disposeForward } from '../../src/engine/model';
import { canReachPrediction, residualHooks, type ActivationReference, type Intervention, type ResidualReplaceIntervention } from '../../src/engine/contracts';
import { planExecution } from '../../src/engine/execution';
import fixture from '../fixtures/residual-reference.json';
import { loadShippedArtifact } from './retrieval-artifact';
import { assertClose } from './numerical';

const verification = loadShippedArtifact('verification'), retrieval = loadShippedArtifact('retrieval');
const cases = fixture.cases as unknown as { name: string; tokenIds: number[]; interventions: Intervention[]; logits: number[][]; stages: Record<string, number[][]> }[];
beforeAll(async () => { await tf.setBackend('cpu'); await tf.ready(); });

it('full execution matches independent PyTorch residual replacement at every hook, alone and with head scaling', async () => {
  expect(fixture.verificationRevision).toBe(verification.manifest.modelRevision);
  const start = tf.memory().numTensors;
  const model = new TransformerModel(verification.manifest.config, verification.artifact.weights);
  let maximum = 0;
  try {
    const plain = model.forward(cases[0].tokenIds);
    const baselineLogits = Array.from(await plain.logits.data());
    disposeForward(plain);
    for (const item of cases) {
      const capture = Object.keys(item.stages).map(key => { const [layer, stage] = key.split(':'); return { layer: Number(layer), stage }; }) as ActivationReference[];
      const result = model.forward(item.tokenIds, item.interventions, capture);
      try {
        maximum = Math.max(maximum, assertClose(await result.logits.data(), item.logits.flat(), `${item.name} logits`));
        for (const [key, expected] of Object.entries(item.stages)) maximum = Math.max(maximum, assertClose(await result.captured[key].data(), expected.flat(), `${item.name} ${key}`));
        for (const patch of item.interventions) if (patch.kind === 'residual.replace') {
          // The hook reports exactly the submitted row.
          const row = (await result.captured[`${patch.layer}:${patch.stage}`].data()).slice(patch.position * 8, patch.position * 8 + 8);
          expect(Array.from(row)).toEqual(Array.from(Float32Array.from(patch.values)));
        }
        if (item.name === 'attention-residual-1-unreachable') expect(Array.from((await result.logits.data()).slice(5 * 16))).toEqual(baselineLogits.slice(5 * 16));
      } finally { disposeForward(result); }
    }
    // Replacing a row with this engine's own value is an exact identity. (The fixture's PyTorch values
    // differ by float32 rounding, so that cross-implementation case is checked within tolerance above.)
    const own = model.forward(cases[0].tokenIds, [], [{ layer: 0, stage: 'attention.residual' }]);
    const row = Array.from((await own.captured['0:attention.residual'].data()).slice(2 * 8, 3 * 8));
    disposeForward(own);
    const identity = model.forward(cases[0].tokenIds, [{ id: 'same', kind: 'residual.replace', layer: 0, stage: 'attention.residual', position: 2, values: row }]);
    try { expect(Array.from(await identity.logits.data())).toEqual(baselineLogits); } finally { disposeForward(identity); }
  } finally { model.dispose(); }
  console.info('Maximum residual-replacement error', maximum);
  expect(tf.memory().numTensors).toBe(start);
});

it('the run engine plans full execution, reports reachability, and inspects replaced runs without the cache', async () => {
  const start = tf.memory().numTensors;
  const engine = new RunEngine(verification.artifact);
  try {
    for (const item of cases) {
      const baseline = await engine.runBaseline(item.tokenIds);
      const position = item.tokenIds.length - 1;
      const result = await engine.compare({ baselineRunId: baseline.runId, interventions: item.interventions, predictionPosition: position });
      expect(result.execution.strategy, item.name).toBe('full');
      expect(result.hasComputationalPath, item.name).toBe(item.interventions.some(patch => canReachPrediction(patch, position, verification.manifest.config, item.tokenIds.length)));
      const logits = await engine.inspectActivation(result.modifiedRunId, { stage: 'logits' }, { position });
      assertClose(logits.values, item.logits[position], `${item.name} engine logits`);
      for (const [key, expected] of Object.entries(item.stages)) {
        const [layer, stage] = key.split(':');
        const slice = await engine.inspectActivation(result.modifiedRunId, { layer: Number(layer), stage } as ActivationReference, { position });
        assertClose(slice.values, expected[position], `${item.name} engine ${key}`);
      }
      engine.releaseRun(baseline.runId);
      // A modified run keeps its replacement after the baseline is released.
      assertClose((await engine.inspectActivation(result.modifiedRunId, { stage: 'logits' }, { position })).values, item.logits[position], `${item.name} replay`);
      engine.releaseRun(result.modifiedRunId);
    }
    expect(planExecution([{ id: 'r', kind: 'residual.replace', layer: 1, stage: 'attention.residual', position: 0, values: Array(8).fill(0) }], true).strategy).toBe('full');
  } finally { engine.dispose(); }
  expect(tf.memory().numTensors).toBe(start);
});

it('rejects invalid replacements before allocating tensors', async () => {
  const engine = new RunEngine(verification.artifact);
  const baseline = await engine.runBaseline([0, 1, 5, 9]);
  const count = tf.memory().numTensors;
  const valid: ResidualReplaceIntervention = { id: 'r', kind: 'residual.replace', layer: 0, stage: 'block.input', position: 1, values: Array(8).fill(.5) };
  const invalid: [Partial<ResidualReplaceIntervention>, string][] = [
    [{ values: Array(7).fill(0) }, '8 finite values'], [{ values: [...Array(7).fill(0), Number.NaN] }, 'finite'],
    [{ values: [...Array(7).fill(0), 1e5] }, '±10000'], [{ stage: 'mlp.output' as never }, 'block.input or attention.residual'],
    [{ position: 4 }, 'outside the input'], [{ layer: 2 }, 'outside the model'],
  ];
  try {
    for (const [patch, message] of invalid)
      await expect(engine.compare({ baselineRunId: baseline.runId, interventions: [{ ...valid, ...patch } as ResidualReplaceIntervention] })).rejects.toThrow(message);
    await expect(engine.compare({ baselineRunId: baseline.runId, interventions: [valid, { ...valid, id: 's' }] })).rejects.toThrow('twice');
    expect(tf.memory().numTensors).toBe(count);
    expect(residualHooks).toEqual(['block.input', 'attention.residual']);
  } finally { engine.dispose(); }
});

it('reproduces the independent retrieval patching sweep with residuals read from the source run', async () => {
  const { patching } = fixture;
  expect(fixture.retrievalRevision).toBe(retrieval.manifest.modelRevision);
  const engine = new RunEngine(retrieval.artifact);
  let maximum = 0;
  try {
    const source = await engine.runBaseline(patching.sourceTokenIds);
    const target = await engine.runBaseline(patching.targetTokenIds);
    for (const entry of patching.sweep) {
      const read = await engine.inspectActivation(source.runId, { layer: entry.layer, stage: entry.stage } as ActivationReference, { position: entry.position });
      const patch: ResidualReplaceIntervention = { id: 'patch', kind: 'residual.replace', layer: entry.layer, stage: entry.stage as ResidualReplaceIntervention['stage'],
        position: entry.position, values: Array.from(read.values) };
      const result = await engine.compare({ baselineRunId: target.runId, interventions: [patch], predictionPosition: patching.predictionPosition });
      const logits = await engine.inspectActivation(result.modifiedRunId, { stage: 'logits' }, { position: patching.predictionPosition });
      maximum = Math.max(maximum, assertClose(logits.values, entry.logits, `${entry.layer}:${entry.stage}@${entry.position}`));
      engine.releaseRun(result.modifiedRunId);
    }
  } finally { engine.dispose(); }
  console.info('Maximum patching-sweep logit error', maximum);
});
