import { useEffect, useRef, useState } from 'react';
import type { ComparisonResult } from '../engine/runs';
import type { EffectTraceReport } from '../engine/trace';
import type { ModelClient } from '../workers/client';

interface Read { key: string; report?: EffectTraceReport; error?: string }
/** Only completed comparisons enter the queue. Navigation never schedules measurement. */
export function useEffectTrace(client: ModelClient | null, comparison: ComparisonResult | null, busy: boolean) {
  const [open, setOpen] = useState(false), [read, setRead] = useState<Read | null>(null);
  const alive = useRef(true), attempt = useRef(''), flight = useRef<object | null>(null);
  const key = comparison ? JSON.stringify([comparison.execution.runtimeGeneration, comparison.baselineRunId, comparison.modifiedRunId, comparison.predictionPosition]) : '';
  const supported = comparison?.interventions.length === 1;
  const enabled = open && !!client && !!comparison && supported && !busy;
  const desired = useRef<{ client: ModelClient; comparison: ComparisonResult; key: string } | null>(null);
  desired.current = enabled ? { client: client!, comparison: comparison!, key } : null;
  const pump = async () => {
    const task = desired.current;
    if (!alive.current || !task || flight.current || attempt.current === task.key) return;
    const token = {}; flight.current = token; attempt.current = task.key;
    let published = false;
    setRead({ key: task.key });
    try {
      const report = await task.client.request('TRACE', { baselineRunId: task.comparison.baselineRunId, modifiedRunId: task.comparison.modifiedRunId, predictionPosition: task.comparison.predictionPosition });
      if (report.baselineRunId !== task.comparison.baselineRunId || report.modifiedRunId !== task.comparison.modifiedRunId || report.runtimeGeneration !== task.comparison.execution.runtimeGeneration)
        throw new Error('The trace belongs to a different completed experiment.');
      if (alive.current && desired.current?.key === task.key && desired.current.client === task.client) { published = true; setRead({ key: task.key, report }); }
    } catch (error) {
      if (alive.current && desired.current?.key === task.key && desired.current.client === task.client) { published = true; setRead({ key: task.key, error: error instanceof Error ? error.message : String(error) }); }
    } finally {
      if (flight.current === token) flight.current = null;
      if (!published && attempt.current === task.key) attempt.current = '';
      if (alive.current && desired.current && desired.current.key !== task.key) void pump();
    }
  };
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; desired.current = null; };
  }, []);
  useEffect(() => {
    if (!enabled) return;
    const timer = setTimeout(() => void pump(), 120);
    return () => clearTimeout(timer);
  }, [enabled, key, client]);
  const current = enabled && read?.key === key ? read : null;
  return { open, setOpen, supported, report: current?.report, error: current?.error, loading: open && supported && !current?.report && !current?.error,
    retry: () => { attempt.current = ''; void pump(); } };
}
