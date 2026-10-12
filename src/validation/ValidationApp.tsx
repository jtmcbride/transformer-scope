import { useRef, useState } from 'react';
import { runHarness } from './harness';
import { defaultSettings, evidenceClasses, validateSettings, type EvidenceClass, type HarnessBackend, type HarnessModel, type HarnessSettings, type ValidationReport } from './report';

const evidenceLabels: Record<EvidenceClass, string> = {
  'unspecified': 'Not specified', 'physical-device': 'Physical mobile device', 'installed-desktop-browser': 'Installed desktop browser',
  'viewport-emulation': 'Desktop viewport or device emulation', 'automation': 'Automated browser (Playwright/CI)',
};

export default function ValidationApp() {
  const host = useRef<HTMLDivElement>(null);
  const [settings, setSettings] = useState<HarnessSettings>(defaultSettings);
  const [evidenceClass, setEvidenceClass] = useState<EvidenceClass>('unspecified');
  const [deviceDescription, setDeviceDescription] = useState('');
  const [notes, setNotes] = useState('');
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const json = report ? JSON.stringify(report, null, 2) + '\n' : '';

  function toggle<T>(list: T[], value: T, on: boolean) { return on ? [...list, value] : list.filter(item => item !== value); }
  async function start() {
    let checked: HarnessSettings;
    try { checked = validateSettings(settings); } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); return; }
    setRunning(true); setError(null); setReport(null); setLog([]); setCopied(false);
    try {
      setReport(await runHarness(checked, { evidenceClass, deviceDescription: deviceDescription.trim(), notes: notes.trim() }, host.current!,
        line => setLog(lines => [...lines, `${new Date().toISOString().slice(11, 19)} ${line}`])));
    } finally { setRunning(false); }
  }
  function download() {
    const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url; link.download = `transformer-microscope-validation-${report!.startedAt.replace(/[:.]/gu, '-')}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function copy() {
    try { await navigator.clipboard.writeText(json); setCopied(true); }
    catch { setError('Clipboard access was refused. Select the JSON below and copy it manually.'); }
  }

  return <main className="validation">
    <header><p className="eyebrow">Diagnostic page</p><h1>Device validation harness</h1>
      <p>Nothing runs until you press Start. The harness opens fresh computation workers and an embedded copy of the application from this deployment. It does not change experiments in other tabs, and results stay on this device until you export them.</p>
      <p>Keep this page visible and the screen awake during the run. Hidden pages are throttled, which invalidates timings; the report records it if that happens.</p></header>
    <form onSubmit={event => { event.preventDefault(); void start(); }}>
      <fieldset disabled={running}><legend>Evidence</legend>
        <label>What is running this page?<select aria-label="Evidence class" value={evidenceClass} onChange={event => setEvidenceClass(event.target.value as EvidenceClass)}>
          {evidenceClasses.map(value => <option key={value} value={value}>{evidenceLabels[value]}</option>)}</select></label>
        <label>Device model and browser, as observed<input aria-label="Device description" value={deviceDescription} maxLength={200} placeholder="e.g. Pixel 6a, Chrome 141, battery 80%, not charging" onChange={event => setDeviceDescription(event.target.value)} /></label>
        <label>Notes<textarea aria-label="Notes" value={notes} maxLength={2000} rows={2} onChange={event => setNotes(event.target.value)} /></label>
      </fieldset>
      <fieldset disabled={running}><legend>Scope</legend>
        <div className="choices">{(['retrieval', 'verification'] as HarnessModel[]).map(model => <label key={model}><input type="checkbox" checked={settings.models.includes(model)}
          onChange={event => setSettings({ ...settings, models: toggle(settings.models, model, event.target.checked) })} />{model === 'retrieval' ? 'Trained retrieval checkpoint' : 'Verification checkpoint'}</label>)}</div>
        <div className="choices">{(['cpu', 'webgl'] as HarnessBackend[]).map(backend => <label key={backend}><input type="checkbox" checked={settings.backends.includes(backend)}
          onChange={event => setSettings({ ...settings, backends: toggle(settings.backends, backend, event.target.checked) })} />{backend === 'cpu' ? 'CPU worker' : 'WebGL worker'}</label>)}</div>
        <div className="choices"><label>Warm-up edits<input type="number" aria-label="Warm-up edits" min={0} max={50} value={settings.warmupSamples} onChange={event => setSettings({ ...settings, warmupSamples: Number(event.target.value) })} /></label>
          <label>Measured edits per layer<input type="number" aria-label="Measured edits" min={1} max={200} value={settings.measuredSamples} onChange={event => setSettings({ ...settings, measuredSamples: Number(event.target.value) })} /></label></div>
        <label className="inline"><input type="checkbox" checked={settings.includeInterface} onChange={event => setSettings({ ...settings, includeInterface: event.target.checked })} />Measure the embedded application interface</label>
      </fieldset>
      <button type="submit" className="primary" disabled={running}>{running ? 'Running…' : 'Start validation'}</button>
    </form>
    {error && <p className="error" role="alert">{error}</p>}
    {(running || log.length > 0) && <section aria-label="Progress"><h2>Progress</h2><ol className="log">{log.map((line, i) => <li key={i}>{line}</li>)}</ol></section>}
    <div ref={host} className="frame-host" />
    {report && <section aria-label="Results">
      <h2>Result: <span data-testid="validation-outcome" className={`outcome ${report.summary.outcome}`}>{report.summary.outcome}</span></h2>
      <p>{report.summary.passed} passed · {report.summary.failed} failed · {report.summary.unsupported} unsupported · {report.summary.skipped} skipped
        {report.evidence.pageHiddenDuringRun && <strong> · The page was hidden during the run; timings are not valid.</strong>}
        {report.abortReason && <strong> · Stopped: {report.abortReason}</strong>}</p>
      <p>Release {report.release.status}: {report.release.detail}</p>
      <table className="check-table"><thead><tr><th>Checkpoint</th><th>Requested</th><th>Check</th><th>Status</th><th>Detail</th></tr></thead>
        <tbody>{report.combinations.flatMap(combination => combination.checks.map(check =>
          <tr key={`${combination.model}-${combination.requestedBackend}-${check.name}`} data-check={`${combination.model}/${combination.requestedBackend}/${check.name}`} className={check.status}>
            <td>{combination.model}</td><td>{combination.requestedBackend}</td><td>{check.name}</td><td>{check.status}</td><td>{check.detail ?? ''}</td></tr>))}</tbody></table>
      <div className="actions"><button type="button" onClick={download}>Download JSON</button><button type="button" onClick={() => void copy()}>{copied ? 'Copied' : 'Copy JSON'}</button></div>
      <details><summary>Raw JSON</summary><pre data-testid="validation-json">{json}</pre></details>
    </section>}
  </main>;
}
