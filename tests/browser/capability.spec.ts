import { expect, test } from '@playwright/test';

test('an unavailable worker WebGL context produces a clear error and CPU recovery', async ({ page }) => {
  // Emulate the actual missing worker capability; CPU inference still uses the real production bundle.
  await page.route('**/model.worker-*.js', async route => {
    const response = await route.fetch();
    const script = await response.text();
    const capability = `if (typeof OffscreenCanvas !== 'undefined') {
      const originalContext = OffscreenCanvas.prototype.getContext;
      OffscreenCanvas.prototype.getContext = function(kind, ...args) {
        if (kind.includes('webgl')) return null;
        return originalContext.call(this, kind, ...args);
      };
    }\n`;
    await route.fulfill({ response, body: capability + script });
  });
  await page.goto('?model=verification&view=details');
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await page.getByLabel('Backend').selectOption('webgl');
  await expect(page.getByRole('alert')).toContainText('WebGL worker unavailable:');
  await page.getByRole('button', { name: 'Load with CPU' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Showing 0.50×' })).toBeVisible();
  await expect(page.getByLabel('Backend')).toHaveValue('cpu');
  await expect(page.getByRole('alert')).toHaveCount(0);
});

// Records the worker's WebGL contexts so the test can make the browser drop them, as a mobile browser may
// when a tab is backgrounded. '__lose' drops them now; '__arm' drops them just after the next comparison starts.
const contextHook = `const __contexts = [];
if (typeof OffscreenCanvas !== 'undefined') {
  const original = OffscreenCanvas.prototype.getContext;
  OffscreenCanvas.prototype.getContext = function(kind, ...args) {
    const context = original.call(this, kind, ...args);
    if (context && String(kind).includes('webgl')) __contexts.push(context);
    return context;
  };
}
let __armed = false;
const __lose = () => __contexts.forEach(context => context.getExtension('WEBGL_lose_context')?.loseContext());
self.addEventListener('message', event => {
  if (event.data === '__lose') { event.stopImmediatePropagation(); __lose(); }
  else if (event.data === '__arm') { event.stopImmediatePropagation(); __armed = true; }
  else if (__armed && event.data?.command === 'COMPARE') { __armed = false; setTimeout(__lose, 0); }
});\n`;

for (const timing of ['idle', 'in-flight'] as const) {
  test(`a lost WebGL context (${timing}) ends the session visibly and recovers on CPU`, async ({ page }, testInfo) => {
    await page.route('**/model.worker-*.js', async route => {
      const response = await route.fetch();
      await route.fulfill({ response, body: contextHook + await response.text() });
    });
    await page.addInitScript(() => {
      const Original = window.Worker, workers: Worker[] = [];
      Object.assign(window, { __workers: workers });
      window.Worker = class extends Original { constructor(url: string | URL, options?: WorkerOptions) { super(url, options); workers.push(this); } };
    });
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('?backend=webgl&view=details&model=retrieval');
    const status = page.locator('.status');
    await expect(status.filter({ hasText: 'Showing 0.00×' }).or(page.getByRole('alert'))).toBeVisible({ timeout: 30_000 });
    const unsupported = await page.getByRole('alert').filter({ hasText: 'WebGL worker unavailable' }).count();
    if (unsupported) { testInfo.annotations.push({ type: 'unsupported-webgl', description: await page.getByRole('alert').innerText() }); return; }
    await expect(status).toHaveText('Showing 0.00×');
    const send = (message: string) => page.evaluate(value => (window as unknown as { __workers: Worker[] }).__workers.at(-1)!.postMessage(value), message);
    if (timing === 'idle') await send('__lose');
    else {
      await send('__arm');
      await page.getByLabel('Exact head strength').fill('0.37');
      await page.getByLabel('Exact head strength').press('Enter');
    }
    const alert = page.getByRole('alert');
    await expect(alert).toContainText('WebGL context lost', { timeout: 10_000 });
    await expect(status).toHaveText('Showing 0.00×');
    // A result requested after the loss is never displayed.
    await expect(page.getByText('Showing 0.37×')).toHaveCount(0);
    await page.getByRole('button', { name: 'Restart on CPU' }).click();
    await expect(page.getByLabel('Backend')).toHaveValue('cpu');
    await expect(status).toHaveText(timing === 'idle' ? 'Showing 0.00×' : 'Showing 0.37×');
    await expect(page.getByRole('alert')).toHaveCount(0);
    expect(errors).toEqual([]);
  });
}
