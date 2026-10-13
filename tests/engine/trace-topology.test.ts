import { expect, it } from 'vitest';
import { canReachPrediction, type AttentionOverrideIntervention, type HeadScaleIntervention, type ModelConfig, type ResidualReplaceIntervention } from '../../src/engine/contracts';
import { traceReferences } from '../../src/engine/trace';
import { traceNodeKey, traceRoute, type TraceAddress } from '../../src/engine/trace-topology';

// Independent dependency graph oracle, with explicit causal attention edges and residual bypasses.
function dependencyGraph(config: ModelConfig, length: number) {
  const next = new Map<string, string[]>(), prior = new Map<string, string[]>();
  const edge = (a: TraceAddress, b: TraceAddress) => {
    const from = traceNodeKey(a), to = traceNodeKey(b);
    next.set(from, [...next.get(from) ?? [], to]); prior.set(to, [...prior.get(to) ?? [], from]);
  };
  for (let position = 0; position < length; position++) {
    edge({ stage: 'embedding.output', position }, { stage: 'block.input', layer: 0, position });
    for (let layer = 0; layer < config.layers; layer++) {
      const at = (stage: TraceAddress['stage'], head?: number): TraceAddress => ({ layer, stage, position, ...(head === undefined ? {} : { head }) });
      edge(at('block.input'), at('attention.normalized')); edge(at('block.input'), at('attention.residual'));
      for (let head = 0; head < config.heads; head++) {
        for (let source = 0; source <= position; source++) {
          edge({ ...at('attention.normalized'), position: source }, at('attention.weights', head));
          edge({ ...at('attention.normalized'), position: source }, at('attention.head.raw', head));
        }
        edge(at('attention.weights', head), at('attention.head.raw', head));
        edge(at('attention.head.raw', head), at('attention.head.modified', head));
        edge(at('attention.head.modified', head), at('attention.projected'));
      }
      edge(at('attention.projected'), at('attention.residual'));
      edge(at('attention.residual'), at('mlp.output')); edge(at('attention.residual'), at('block.output'));
      edge(at('mlp.output'), at('block.output'));
      edge(at('block.output'), layer + 1 < config.layers ? { stage: 'block.input', layer: layer + 1, position } : { stage: 'final.normalized', position });
    }
    edge({ stage: 'final.normalized', position }, { stage: 'logits', position });
  }
  const visit = (roots: string[], edges: Map<string, string[]>) => {
    const found = new Set(roots), queue = [...roots];
    for (let i = 0; i < queue.length; i++) for (const key of edges.get(queue[i]) ?? []) if (!found.has(key)) { found.add(key); queue.push(key); }
    return found;
  };
  return { next, prior, visit };
}

it('position-aware routes agree with the dependency graph, including identities, bypasses and cross-position attention', () => {
  for (const layers of [1, 2, 3]) {
    const config: ModelConfig = { layers, heads: 2, modelDim: 8, mlpDim: 16, contextLength: 4, vocabularySize: 16, layerNormEpsilon: 1e-5 };
    const length = 4, graph = dependencyGraph(config, length);
    for (let layer = 0; layer < layers; layer++) for (const positions of [[], [0], [2], [3], [0, 2], 'all'] as const) for (const factor of [0, 1]) {
      const patch: HeadScaleIntervention = { id: 'route', kind: 'head.scale', layer, head: 0, factor,
        tokens: positions === 'all' ? { kind: 'all' } : { kind: 'indices', indices: [...positions] } };
      const roots = factor === 1 ? [] : (positions === 'all' ? [0, 1, 2, 3] : positions).map(position => traceNodeKey({ stage: 'attention.head.modified', layer, head: 0, position }));
      const downstream = graph.visit(roots, graph.next);
      for (let prediction = 0; prediction < length; prediction++) {
        const end = traceNodeKey({ stage: 'logits', position: prediction }), ancestors = graph.visit([end], graph.prior);
        expect(downstream.has(end)).toBe(canReachPrediction(patch, prediction, config, length));
        for (const reference of traceReferences(config)) for (let position = 0; position < length; position++) {
          const headed = ['attention.weights', 'attention.head.raw', 'attention.head.modified'].includes(reference.stage);
          for (let head = 0; head < (headed ? config.heads : 1); head++) {
            const node = { ...reference, position, ...(headed ? { head } : {}) }, key = traceNodeKey(node);
            expect(traceRoute(node, patch, config, length, prediction), key).toBe(!downstream.has(key) ? 'unaffected' : ancestors.has(key) ? 'prediction' : 'local');
          }
        }
      }
    }
  }
});

it('redirect routes start at the query attention row, rather than at the source being read', () => {
  for (const layers of [1, 2, 3]) {
    const config: ModelConfig = { layers, heads: 2, modelDim: 8, mlpDim: 16, contextLength: 4, vocabularySize: 16, layerNormEpsilon: 1e-5 };
    const length = 4, graph = dependencyGraph(config, length);
    for (let layer = 0; layer < layers; layer++) for (let query = 0; query < length; query++) for (let head = 0; head < config.heads; head++) {
      const patch: AttentionOverrideIntervention = { id: 'redirect', kind: 'attention.override', layer, head, query, weights: [1, 0, 0, 0] };
      const downstream = graph.visit([traceNodeKey({ stage: 'attention.weights', layer, head, position: query })], graph.next);
      for (let prediction = 0; prediction < length; prediction++) {
        const end = traceNodeKey({ stage: 'logits', position: prediction }), ancestors = graph.visit([end], graph.prior);
        expect(downstream.has(end)).toBe(canReachPrediction(patch, prediction, config, length));
        for (const reference of traceReferences(config)) for (let position = 0; position < length; position++) {
          const headed = ['attention.weights', 'attention.head.raw', 'attention.head.modified'].includes(reference.stage);
          for (let h = 0; h < (headed ? config.heads : 1); h++) {
            const node = { ...reference, position, ...(headed ? { head: h } : {}) }, key = traceNodeKey(node);
            expect(traceRoute(node, patch, config, length, prediction), key).toBe(!downstream.has(key) ? 'unaffected' : ancestors.has(key) ? 'prediction' : 'local');
          }
        }
        expect(traceRoute({ stage: 'attention.scores', layer, head, position: query }, patch, config, length, prediction)).toBe('unaffected');
      }
    }
  }
});

it('residual routes begin at the copied junction and mix positions only through subsequent attention', () => {
  for (const layers of [1, 2, 3]) {
    const config: ModelConfig = { layers, heads: 2, modelDim: 8, mlpDim: 16, contextLength: 4, vocabularySize: 16, layerNormEpsilon: 1e-5 };
    const length = 4, graph = dependencyGraph(config, length);
    for (let layer = 0; layer < layers; layer++) for (let root = 0; root < length; root++) for (const stage of ['block.input', 'attention.residual'] as const) {
      const patch: ResidualReplaceIntervention = { id: 'copy', kind: 'residual.replace', layer, stage, position: root, values: Array(8).fill(0) };
      const downstream = graph.visit([traceNodeKey({ stage, layer, position: root })], graph.next);
      for (let prediction = 0; prediction < length; prediction++) {
        const end = traceNodeKey({ stage: 'logits', position: prediction }), ancestors = graph.visit([end], graph.prior);
        expect(downstream.has(end)).toBe(canReachPrediction(patch, prediction, config, length));
        for (const reference of traceReferences(config)) for (let position = 0; position < length; position++) {
          const headed = ['attention.weights', 'attention.head.raw', 'attention.head.modified'].includes(reference.stage);
          for (let h = 0; h < (headed ? config.heads : 1); h++) {
            const node = { ...reference, position, ...(headed ? { head: h } : {}) }, key = traceNodeKey(node);
            expect(traceRoute(node, patch, config, length, prediction), `${stage} ${root} → ${prediction}, ${key}`).toBe(!downstream.has(key) ? 'unaffected' : ancestors.has(key) ? 'prediction' : 'local');
          }
        }
      }
    }
  }
});
