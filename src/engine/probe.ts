import * as tf from '@tensorflow/tfjs-core';
import { TransformerModel, disposeForward } from './model';
import { EngineError, type ActivationReference, type HeadScaleIntervention, type ModelArtifact, type TensorRecord } from './contracts';

export interface ProbeFixture {
  cases: { name: string; tokenIds: number[]; interventions: HeadScaleIntervention[]; activations: Record<string, TensorRecord> }[];
  training: { tokenIds: number[]; targets: number[]; loss: number;
    optimizer: { learningRate: number; beta1: number; beta2: number; epsilon: number };
    gradients: Record<string, TensorRecord>; updatedWeights: Record<string, TensorRecord> };
}
export interface ProbeReport {
  backend: string; elapsedMs: number; forwardMaximumError: number; gradientMaximumError: number;
  optimizerMaximumError: number; loss: number; retainedTensorDelta: number;
  float32Textures: boolean | null;
  webglPacking: boolean | null;
}
function check(actual: ArrayLike<number>, expected: ArrayLike<number>, atol: number, rtol: number, name: string): number {
  if (actual.length !== expected.length) throw new EngineError('NUMERICAL_FAILURE', `Probe shape mismatch: ${name}`);
  let maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(actual[i]) || error > atol + rtol * Math.abs(expected[i]))
      throw new EngineError('NUMERICAL_FAILURE', `Probe mismatch at ${name}[${i}]: error ${error}`);
    maximum = Math.max(maximum, error);
  }
  return maximum;
}
export async function runProbe(artifact: ModelArtifact, fixture: ProbeFixture): Promise<ProbeReport> {
  const count = tf.memory().numTensors;
  const start = performance.now();
  const model = new TransformerModel(artifact.manifest.config, artifact.weights);
  const o = fixture.training.optimizer;
  const optimizer = tf.train.adam(o.learningRate, o.beta1, o.beta2, o.epsilon);
  let forwardMaximumError = 0, gradientMaximumError = 0, optimizerMaximumError = 0, loss = 0;
  try {
    for (const example of fixture.cases) {
      const capture = Object.keys(example.activations).map(key => {
        const parts = key.split(':');
        return parts.length === 1 ? { stage: parts[0] } : { layer: Number(parts[0]), stage: parts[1] };
      }) as ActivationReference[];
      const result = model.forward(example.tokenIds, example.interventions, capture);
      try {
        for (const [key, expected] of Object.entries(example.activations))
          forwardMaximumError = Math.max(forwardMaximumError, check(await result.captured[key].data(), expected.values, 2e-5, 1e-5, key));
      } finally { disposeForward(result); }
    }
    const gradient = tf.variableGrads(() => model.loss(fixture.training.tokenIds, fixture.training.targets), Object.values(model.weights));
    try {
      loss = (await gradient.value.data())[0];
      check([loss], [fixture.training.loss], 2e-5, 1e-5, 'loss');
      const gradientValues: Record<string, ArrayLike<number>> = {};
      for (const [name, variable] of Object.entries(model.weights)) gradientValues[name] = await gradient.grads[variable.name].data();
      const invalid = Object.entries(gradientValues).map(([name, values]) => ({ name, count: Array.from(values).filter(value => !Number.isFinite(value)).length })).filter(entry => entry.count);
      if (invalid.length) throw new EngineError('NUMERICAL_FAILURE', `Non-finite gradients: ${invalid.map(entry => `${entry.name} (${entry.count})`).join(', ')}`);
      for (const [name, values] of Object.entries(gradientValues))
        gradientMaximumError = Math.max(gradientMaximumError, check(values, fixture.training.gradients[name].values, 3e-5, 3e-4, `gradient/${name}`));
      optimizer.applyGradients(Object.entries(gradient.grads).map(([name, tensor]) => ({ name, tensor })));
      for (const [name, variable] of Object.entries(model.weights))
        optimizerMaximumError = Math.max(optimizerMaximumError, check(await variable.data(), fixture.training.updatedWeights[name].values, 3e-5, 1e-5, `updated/${name}`));
    } finally { gradient.value.dispose(); Object.values(gradient.grads).forEach(g => g.dispose()); }
  } finally { optimizer.dispose(); model.dispose(); }
  return { backend: tf.getBackend(), elapsedMs: performance.now() - start, forwardMaximumError, gradientMaximumError,
    optimizerMaximumError, loss, retainedTensorDelta: tf.memory().numTensors - count,
    float32Textures: tf.getBackend() === 'webgl' ? tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED') : null,
    webglPacking: tf.getBackend() === 'webgl' ? tf.env().getBool('WEBGL_PACK') : null };
}
