import { useState } from 'react';
import { StoryHeadScan, type StoryScanProps } from './StoryHeadScan';
import { StoryHeadPairScan } from './StoryHeadPairScan';
import { useStoryWorkspace } from './StoryWorkspace';

export function StoryAnalysis(props: StoryScanProps & { targetTokenId: number; onTargetChange: (id: number) => void; targeted: boolean }) {
  const workspace = useStoryWorkspace()!;
  const [visited, setVisited] = useState(workspace.analysis);
  function toggle(kind: 'heads' | 'pairs', open: boolean) {
    if (open) setVisited(old => old[kind] ? old : { ...old, [kind]: true });
    workspace.setAnalysisOpen(kind, open);
  }
  return <section className="story-analysis" aria-label="Advanced token analysis">
    <div><p className="eyebrow">Explore more changes</p><h4>Find heads worth changing</h4><p>Open a scan to rank possible changes, then use a result in the branch controls above.</p></div>
    <details className="story-analysis-heads" open={workspace.analysis.heads} onToggle={event => toggle('heads', event.currentTarget.open)}>
      <summary>Scan individual heads <span>Mute, restore, or amplify one head</span></summary>
      {(visited.heads || workspace.analysis.heads) && <StoryHeadScan {...props} active={workspace.analysis.heads} />}
    </details>
    <details className="story-analysis-pairs" open={workspace.analysis.pairs} onToggle={event => toggle('pairs', event.currentTarget.open)}>
      <summary>Scan head pairs <span>Rank combinations and explore an interaction map</span></summary>
      {(visited.pairs || workspace.analysis.pairs) && <StoryHeadPairScan key={props.targetTokenId} {...props} active={workspace.analysis.pairs} />}
    </details>
  </section>;
}
