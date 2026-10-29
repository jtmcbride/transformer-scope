import type { Page } from '@playwright/test';

/** Enter advanced analysis explicitly, as a user would, before exercising its controls. */
export async function openStoryAnalysis(page: Page, kinds: ('heads' | 'pairs')[] = ['heads', 'pairs']) {
  for (const kind of kinds) {
    const panel = page.locator(`.story-analysis-${kind}`);
    if (await panel.getAttribute('open') === null) await panel.locator('summary').first().click();
  }
}
