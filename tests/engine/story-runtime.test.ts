import { expect, it, vi } from 'vitest';
import { withStoryPrefix } from '../../src/workers/story-runtime';
import type { ModelClient } from '../../src/workers/client';
const edit = { id: 'test', kind: 'head.scale' as const, layer: 0, head: 0, factor: 0, tokens: { kind: 'all' as const } };
function fake(fail = false) {
  let modified = 0;
  const worker = { endReason: null as Error | null, request: vi.fn(async (command: string, payload: any) => {
    if (command === 'BASELINE') return { runId: 'baseline' };
    if (command === 'COMPARE') return { modifiedRunId: `changed-${++modified}` };
    if (command === 'INSPECT') { if (fail && payload.runId !== 'baseline') throw new Error('inspection failed'); return { values: [1,2,3] }; }
    return null;
  }) };
  return { worker, client: worker as unknown as ModelClient };
}
it('a scan shares exactly one immutable baseline and releases every completed trial', async () => {
  const { worker, client } = fake(), prefix = [0,3];
  await withStoryPrefix(client, prefix, async predict => {
    expect(await predict([0,3], [])).toEqual([1,2,3]);
    for (const factor of [0,1,2]) expect(await predict([0,3], [{ ...edit, factor }])).toEqual([1,2,3]);
    expect(worker.request.mock.calls.filter(([command,payload]) => command === 'RELEASE' && payload.runId === 'baseline')).toHaveLength(0);
  });
  expect(prefix).toEqual([0,3]);
  expect(worker.request.mock.calls.filter(([command]) => command === 'BASELINE')).toHaveLength(1);
  expect(worker.request.mock.calls.filter(([command]) => command === 'RELEASE').map(([,payload]) => payload.runId)).toEqual(['changed-1','changed-2','changed-3','baseline']);
});
it('failed inspections and mismatched prefixes release the retained baseline; terminated workers need no release RPC', async () => {
  const failed = fake(true);
  await expect(withStoryPrefix(failed.client, [0,3], predict => predict([0,3], [edit]))).rejects.toThrow('inspection failed');
  expect(failed.worker.request.mock.calls.filter(([command]) => command === 'RELEASE').map(([,payload]) => payload.runId)).toEqual(['changed-1','baseline']);
  const mismatch = fake();
  await expect(withStoryPrefix(mismatch.client, [0,3], predict => predict([0,4], []))).rejects.toThrow('same starting text');
  expect(mismatch.worker.request.mock.calls.map(([command]) => command)).toEqual(['BASELINE','RELEASE']);
  const stopped = fake();
  await withStoryPrefix(stopped.client, [0,3], async () => { stopped.worker.endReason = new Error('terminated'); });
  expect(stopped.worker.request.mock.calls.map(([command]) => command)).toEqual(['BASELINE']);
});
