import { useEffect, useMemo, useRef, useState } from 'react';
import { BpeTokenizer, type BpeDocument } from '../engine/bpe';
import type { ModelManifest } from '../engine/contracts';
import type { ComparisonResult } from '../engine/runs';
import { tokenLabel } from '../engine/tokenizer';
import { chooseToken, drawAt, selectionEdits, type StorySelection } from '../continuation';
import type { FlowNode } from '../flow';
import { ModelClient } from '../workers/client';
import { VisualFlow } from './VisualFlow';
import { NodeInspector } from './NodeInspector';
import { useStoryWorkspace } from './StoryWorkspace';
import { StoryFork } from './StoryFork';

interface Props {
  selection: StorySelection; manifest: ModelManifest; tokenizer: BpeDocument; count: number;
  onClose: () => void; onNavigate: (index: number) => void; onForkStart: () => void;
  startStep?: number; depth?: number; parentId?: string;
}
export function StoryTokenInspector({ selection, manifest, tokenizer, count, onClose, onNavigate, onForkStart, startStep = 1, depth = 0, parentId }: Props) {
  const workspace = useStoryWorkspace();
  const currentNode = workspace?.data.nodes.find(n => n.id === workspace.data.route[depth]);
  const initialChild = JSON.stringify(currentNode?.plan.source) === JSON.stringify(selection) && workspace?.data.route[depth + 1];
  const heading = useRef<HTMLHeadingElement>(null);
  const [childActive, setChildActive] = useState(!!initialChild);
  const edits = selectionEdits(selection);
  const initialHead = edits.at(-1);
  const [node, setNode] = useState<FlowNode>({ kind: 'readout' });
  const [retry, setRetry] = useState(0);
  const [result, setResult] = useState<{ client: ModelClient; comparison: ComparisonResult } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attention, setAttention] = useState<{ key: string; values: number[]; error?: string } | null>(null);
  const position = selection.prefix.length - 1;
  const encoder = useMemo(() => new BpeTokenizer(tokenizer), [tokenizer]);
  const branch = depth ? selection.branch === 'original' ? 'Keep source settings' : 'With head change' : selection.branch === 'original' ? 'Original' : 'Edited';
  const label = tokenLabel(tokenizer, selection.choice.tokenId);
  const layer = node.kind === 'head' || node.kind === 'stage' ? node.layer : initialHead?.layer ?? 0;
  const head = node.kind === 'head' ? node.head : initialHead?.head ?? 0;
  const attentionKey = `${result?.comparison.modifiedRunId}:${layer}:${head}`;

  useEffect(() => { heading.current?.focus({ preventScroll: true }); heading.current?.scrollIntoView({ block: 'start', behavior: 'instant' }); }, []);
  useEffect(() => {
    setResult(null); setError(null);
    if (childActive) return;
    let current = true;
    const client = new ModelClient();
    const fail = (failure: unknown) => {
      if (current) { setError(failure instanceof Error ? failure.message : String(failure)); setResult(null); client.dispose(); }
    };
    client.onEnded = fail;
    void (async () => {
      const loaded = await client.request('LOAD', { manifestUrl: new URL(`${import.meta.env.BASE_URL}models/tinystories/manifest.json`, location.origin).href, backend: 'cpu' });
      if (!current) return;
      if (loaded.manifest.modelRevision !== manifest.modelRevision || loaded.manifest.weightsSha256 !== manifest.weightsSha256 || loaded.manifest.tokenizerSha256 !== manifest.tokenizerSha256) {
        throw new Error('The recorded checkpoint is no longer available. Reload the page.');
      }
      const baseline = await client.request('BASELINE', { tokenIds: selection.prefix });
      if (!current) return;
      const comparison = await client.request('COMPARE', { baselineRunId: baseline.runId, interventions: selectionEdits(selection), predictionPosition: position, targetTokenId: selection.choice.tokenId, topK: 8 });
      if (!current) return;
      const logits = await client.request('INSPECT', { runId: comparison.modifiedRunId, reference: { stage: 'logits' }, selection: { position } });
      const replay = chooseToken(logits.values, selection.settings, drawAt(selection.settings.seed, selection.step - 1));
      if (replay.tokenId !== selection.choice.tokenId || Math.abs(replay.probability - selection.choice.probability) > 1e-6) {
        throw new Error('This computation did not reproduce the recorded token. Close and retry the inspection.');
      }
      if (current) setResult({ client, comparison });
    })().catch(fail);
    // One inspected step owns one worker. Replacing/closing the step discards all runs and pending reads.
    return () => { current = false; client.dispose(); };
  }, [selection, manifest, position, retry, childActive]);

  useEffect(() => {
    if (!result || childActive) return;
    let current = true;
    void result.client.request('INSPECT', { runId: result.comparison.modifiedRunId, reference: { stage: 'attention.weights', layer }, selection: { head } })
      .then(slice => { if (current) setAttention({ key: attentionKey, values: Array.from(slice.values) }); })
      .catch(failure => { if (current) setAttention({ key: attentionKey, values: [], error: failure instanceof Error ? failure.message : String(failure) }); });
    return () => { current = false; };
  }, [result, layer, head, attentionKey, childActive]);
  const target = result?.comparison.displayTokens.find(row => row.tokenId === selection.choice.tokenId);
  const weights = attention?.key === attentionKey ? attention : null;

  return <section className={childActive ? "inspection-ancestor" : "story-token-inspector"} aria-label={childActive ? undefined : "Generated token computation"} aria-busy={!childActive && !result && !error}>
    {!childActive && <>
    <div className="token-inspection-heading"><div><p className="eyebrow">{depth > 0 && `Comparison ${depth} · `}{branch} story · step {selection.step}</p><h2 ref={heading} tabIndex={-1}>How the model chose <span>{label}</span></h2></div>
      <div className="token-inspection-navigation" role="group" aria-label="Generated token navigation">
        <button type="button" disabled={selection.step <= startStep} onClick={() => onNavigate(selection.step - startStep - 1)}>← Previous token</button>
        <button type="button" disabled={selection.step >= startStep + count - 1} onClick={() => onNavigate(selection.step - startStep + 1)}>Next token →</button>
        <button type="button" onClick={onClose}>{depth ? 'Back to previous comparison' : 'Close token inspection'}</button>
      </div>
    </div>
    <p>This flow uses the {selection.prefix.length} input tokens <strong>before</strong> step {selection.step}, including the beginning token. The selected token is the output, not part of its own input.</p>
    <p className="inspection-prefix" aria-label="Context before selected token">{encoder.preview(selection.prefix) || '(Beginning of story)'}</p>
    <p className="hint">{edits.length ? `Captured ${edits.length === 1 ? 'edit' : 'edits'}: ${edits.map(edit => `Layer ${edit.layer + 1} · Head ${edit.head + 1} · ${edit.factor.toFixed(2)}×`).join('; ')} at all positions. Both diagram runs use this branch’s context; “without edit” is not the separately generated reference story.` : 'Original model with no intervention. Baseline and modified vectors are identical.'}</p>
    <p className="hint">{selection.settings.temperature === 0 ? 'Chosen by most-likely-token selection.' : `Chosen by seeded sampling · temperature ${selection.settings.temperature} · top-p ${selection.settings.topP} · seed ${selection.settings.seed}. A sampled token need not be the most likely token.`} Probabilities below are the raw model probabilities before sampling.</p>
    {error ? <div role="alert" className="error">{error} <button type="button" onClick={() => setRetry(value => value + 1)}>Retry token inspection</button></div> : !result ? <p role="status">Recomputing this recorded step…</p> : <>
      <div className="selected-token-result" data-testid="selected-token-result"><span>Generated token <strong>{label}</strong></span><span>Model probability <strong data-testid="selected-token-probability">{(target!.probability * 100).toFixed(4)}%</strong></span><span>Recorded choice reproduced</span></div>
      <VisualFlow config={manifest.config} tokens={selection.prefix.map(id => tokenLabel(tokenizer, id))} selected={node} comparison={result.comparison}
        attention={weights?.values ?? []} layer={layer} head={head} position={position} pending={false} ended={false} disabled={false}
        onSelect={setNode} history={{ edited: edits.length > 0 }} />
    </>}
    </>}
    <StoryFork source={selection} manifest={manifest} tokenizer={tokenizer} head={{ layer, head }} disabled={!result || childActive} onStart={onForkStart} onInspectChange={setChildActive} depth={depth} parentId={parentId} analysisClient={result?.client} />
    {!childActive && result && <>
      {weights?.error && <p role="alert" className="error">Could not read attention: {weights.error}</p>}
      <div className="historical-node"><p className="eyebrow">Inside the selected stage</p>
        <p className="hint">Measured at the final input position, {position}. {edits.length ? 'Baseline = without edits on this same context. Modified = with the captured edits.' : 'Both columns describe the original model.'}</p>
        <NodeInspector node={node} client={result.client} baselineRunId={result.comparison.baselineRunId} modifiedRunId={result.comparison.modifiedRunId}
          position={position} symbol={tokenLabel(tokenizer, selection.prefix[position])} allowVector />
      </div>
    </>}
  </section>;
}
