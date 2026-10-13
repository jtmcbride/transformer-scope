export type FlowStage = 'block.input' | 'attention.normalized' | 'attention.projected' | 'attention.residual' | 'mlp.output' | 'block.output';
export type FlowNode = { kind: 'embedding' } | { kind: 'readout' } | { kind: 'head'; layer: number; head: number; part?: 'attention' } | { kind: 'stage'; layer: number; stage: FlowStage };
export const nodeKey = (node: FlowNode) => node.kind === 'head' ? `head:${node.layer}:${node.head}${node.part === 'attention' ? ':attention' : ''}` : node.kind === 'stage' ? `${node.layer}:${node.stage}` : node.kind;
export const stageNames: Record<FlowStage, string> = {
  'block.input': 'Layer input', 'attention.normalized': 'Normalize for attention', 'attention.projected': 'Join heads + project',
  'attention.residual': 'Residual after attention', 'mlp.output': 'Normalize + MLP', 'block.output': 'Layer output',
};
