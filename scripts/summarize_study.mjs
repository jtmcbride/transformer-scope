// Summarizes exported study sessions for docs/educational-findings-template.md.
// Usage: node scripts/summarize_study.mjs [--codes codes.json] session.json... > findings-tables.md
// codes.json maps participant IDs to human-assigned probe codes: { "P01": { "C1": "C", "C2": "P", ... } }.
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { observedDirection, observedMagnitude, probes, scoreTask, tasks, validateSession } from '../src/study/session.ts';

const probeIds = probes.map(probe => probe.id);
const gateProbes = ['C1', 'C2', 'C3'];
const cell = value => String(value ?? '').replace(/\|/gu, '\\|').replace(/\s+/gu, ' ').trim();
const table = (header, rows) => [`| ${header.join(' | ')} |`, `| ${header.map(() => '---').join(' | ')} |`, ...rows.map(row => `| ${row.map(cell).join(' | ')} |`)].join('\n');

export function summarizeStudy(rawSessions, codes = {}) {
  const sessions = rawSessions.map(validateSession).sort((a, b) => a.participant.id.localeCompare(b.participant.id));
  const ids = sessions.map(session => session.participant.id);
  if (new Set(ids).size !== ids.length) throw new Error('Each participant ID must appear once.');
  for (const [id, entry] of Object.entries(codes)) {
    if (!ids.includes(id)) throw new Error(`Codes refer to unknown participant ${id}.`);
    for (const [probe, code] of Object.entries(entry))
      if (!probeIds.includes(probe) || !['C', 'P', 'M', 'N'].includes(code)) throw new Error(`Invalid code ${probe}=${code} for ${id}; use C, P, M, or N.`);
  }
  const n = sessions.length;
  const commits = [...new Set(sessions.map(session => session.release.commit ?? 'unidentified'))];
  const coded = sessions.every(session => probeIds.every(probe => codes[session.participant.id]?.[probe]));
  const code = (session, probe) => {
    const value = codes[session.participant.id]?.[probe];
    return value ? `${value}${session.probes[probe].hint ? 'h' : ''}` : session.probes[probe].hint ? '(uncoded, hint)' : '(uncoded)';
  };
  const lines = [`Generated from ${n} session record(s). Deployed commit(s): ${commits.join(', ')}.`, ''];
  lines.push('## Participants', '', table(['ID', 'Experience band', 'Device / browser', 'Commit', 'Started (UTC)'],
    sessions.map(session => [session.participant.id, session.participant.experienceBand, session.participant.device, (session.release.commit ?? 'unidentified').slice(0, 7), session.startedAt])), '');
  lines.push('## Predictions', '', 'Observed categories come from the live result shown when the prediction was revealed. "Same/none" means exactly zero distribution change; "tiny" is a non-zero change under 1 percentage point.', '',
    table(['ID', 'Task', 'P1 direction', 'P2 magnitude', 'P3 reasoning', 'P4 attention changes?', 'Observed', 'Direction correct', 'Magnitude correct', 'R1/R2'],
      sessions.flatMap(session => session.tasks.map((task, index) => {
        const score = scoreTask(task);
        return [session.participant.id, `${index + 1} ${task.recipeId}`, task.prediction.direction, task.prediction.magnitude, task.prediction.reasoning,
          `${task.prediction.attention}${task.prediction.attentionReason ? `: ${task.prediction.attentionReason}` : ''}`,
          `${observedDirection(task.observation)}, ${observedMagnitude(task.observation)}`, score.direction ? 'yes' : 'no', score.magnitude ? 'yes' : 'no',
          [task.followUp.match, task.followUp.explanation].filter(Boolean).join(' / ')];
      }))), '');
  lines.push('## Concept probes', '', 'Codes: C correct, P partial, M misconception, N no answer; "h" marks an answer after a facilitator hint.', '',
    table(['ID', ...probeIds, 'Verbatim answers'], sessions.map(session => [session.participant.id, ...probeIds.map(probe => code(session, probe)),
      probeIds.map(probe => `${probe}: ${session.probes[probe].answer}`).join(' / ')])), '');
  const needed = Math.ceil(.8 * n);
  const gateRows = gateProbes.map(probe => {
    const correct = sessions.filter(session => codes[session.participant.id]?.[probe] === 'C' && !session.probes[probe].hint).length;
    const hinted = sessions.filter(session => codes[session.participant.id]?.[probe] === 'C' && session.probes[probe].hint).length;
    return [`${probe} correct without a hint`, `≥ ${needed} of ${n} (and n ≥ 5)`, coded ? `${correct} of ${n} (${hinted} more after a hint)` : 'not evaluated: codes missing',
      !coded ? 'not evaluated' : n >= 5 && correct >= needed ? 'yes' : 'no'];
  });
  const misconceptions = probeIds.map(probe => [probe, sessions.filter(session => codes[session.participant.id]?.[probe] === 'M').map(session => session.participant.id).join(', ') || 'none']);
  lines.push('## Summary against the gate', '', table(['Criterion', 'Required', 'Observed', 'Met?'], [
    ...gateRows,
    ...tasks.map((task, index) => [`Task ${index + 1} (${task.recipeId}) direction predicted correctly`, 'reported only',
      `${sessions.filter(session => scoreTask(session.tasks[index]).direction).length} of ${n}`, 'n/a']),
    ['Predicted attention weights unchanged, Task 1', 'reported only', `${sessions.filter(session => session.tasks[0].prediction.attention === 'no').length} of ${n}`, 'n/a'],
  ]), '', '### Misconceptions by probe', '', table(['Probe', 'Participants coded M'], misconceptions), '');
  if (n < 5) lines.push(`Only ${n} session(s): the gate requires at least five participants.`, '');
  return lines.join('\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const codesAt = args.indexOf('--codes');
  const codes = codesAt >= 0 ? JSON.parse(readFileSync(args.splice(codesAt, 2)[1], 'utf8')) : {};
  if (!args.length) { console.error('Usage: node scripts/summarize_study.mjs [--codes codes.json] session.json...'); process.exit(2); }
  process.stdout.write(summarizeStudy(args.map(file => JSON.parse(readFileSync(file, 'utf8'))), codes) + '\n');
}
