/** Drives the real application in a same-origin frame, for the opt-in validation and study pages. */

export interface Frame { frame: HTMLIFrameElement; win: Window & typeof globalThis; doc: Document; errors: string[] }
const message = (error: unknown) => error instanceof Error ? error.message : String(error);

export function statusText(doc: Document) { return doc.querySelector('.status')?.textContent ?? ''; }
export function button(doc: Document, text: string) {
  return [...doc.querySelectorAll('button')].find(element => element.textContent?.trim() === text) ?? null;
}
export function settled(doc: Document) {
  const exported = button(doc, 'Export experiment');
  return statusText(doc).startsWith('Showing') && !!exported && !exported.disabled;
}
/** Resolves when the predicate holds, timed in the frame's own clock (its navigation-start time origin). */
export function waitFor(target: Frame, predicate: () => boolean, label: string, timeoutMs = 60_000): Promise<{ at: number; alreadyTrue: boolean }> {
  if (predicate()) return Promise.resolve({ at: target.win.performance.now(), alreadyTrue: true });
  return new Promise((resolve, reject) => {
    const observer = new target.win.MutationObserver(() => {
      if (!predicate()) return;
      observer.disconnect(); clearTimeout(timer); resolve({ at: target.win.performance.now(), alreadyTrue: false });
    });
    const timer = setTimeout(() => { observer.disconnect(); reject(new Error(`Timed out waiting for ${label}. Status: "${statusText(target.doc)}".`)); }, timeoutMs);
    observer.observe(target.doc, { childList: true, subtree: true, characterData: true, attributes: true });
  });
}
export function select(doc: Document, label: string) {
  const element = doc.querySelector<HTMLSelectElement>(`select[aria-label="${label}"]`);
  if (!element) throw new Error(`The ${label} control is missing.`);
  return element;
}
/** Changes a select through React's change event and waits for the resulting completed result. */
export async function change(target: Frame, element: HTMLSelectElement, value: string, started = 'Computing') {
  if (element.value === value) return;
  const computing = waitFor(target, () => statusText(target.doc).startsWith(started), 'computation to start');
  const setter = Object.getOwnPropertyDescriptor(target.win.HTMLSelectElement.prototype, 'value')!.set!;
  setter.call(element, value); element.dispatchEvent(new target.win.Event('change', { bubbles: true }));
  await computing; await waitFor(target, () => settled(target.doc), 'the computed result');
}

export async function openFrame(host: HTMLElement, url: URL, title: string, className = 'measured-frame'): Promise<Frame> {
  const frame = document.createElement('iframe');
  frame.title = title;
  frame.className = className;
  const loaded = new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('The application page did not load within 60 s.')), 60_000);
    frame.addEventListener('load', () => { clearTimeout(timer); resolve(); }, { once: true });
  });
  frame.src = url.href;
  host.replaceChildren(frame);
  await loaded;
  const win = frame.contentWindow as Window & typeof globalThis, doc = frame.contentDocument!;
  const errors: string[] = [];
  win.addEventListener('error', event => errors.push(event.message));
  win.addEventListener('unhandledrejection', event => errors.push(message(event.reason)));
  return { frame, win, doc, errors };
}
