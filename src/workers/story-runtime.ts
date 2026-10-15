import type { HeadScaleIntervention, ModelManifest } from '../engine/contracts';
import type { ModelClient } from './client';

export async function loadStoryModel(client: ModelClient, manifest: ModelManifest) {
  const loaded = await client.request('LOAD', { manifestUrl: new URL(`${import.meta.env.BASE_URL}models/tinystories/manifest.json`, location.origin).href, backend: 'cpu' });
  if (loaded.manifest.modelRevision !== manifest.modelRevision || loaded.manifest.weightsSha256 !== manifest.weightsSha256 || loaded.manifest.tokenizerSha256 !== manifest.tokenizerSha256) {
    throw new Error('The recorded checkpoint is no longer available. Reload the page.');
  }
}

/** Each prediction owns and releases its temporary descriptors, including failed inspections. */
export async function predictStory(client: ModelClient, tokenIds: number[], edits: HeadScaleIntervention[] = []) {
  const baseline = await client.request('BASELINE', { tokenIds });
  let modified: string | undefined;
  try {
    if (edits.length) modified = (await client.request('COMPARE', { baselineRunId: baseline.runId, interventions: edits, predictionPosition: tokenIds.length - 1 })).modifiedRunId;
    return (await client.request('INSPECT', { runId: modified ?? baseline.runId, reference: { stage: 'logits' }, selection: { position: tokenIds.length - 1 } })).values;
  } finally {
    if (!client.endReason) {
      if (modified) await client.request('RELEASE', { runId: modified });
      await client.request('RELEASE', { runId: baseline.runId });
    }
  }
}
