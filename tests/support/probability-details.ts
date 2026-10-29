import type { Locator } from '@playwright/test';

export async function openProbabilityDetails(report: Locator) {
  const details = report.locator('.branch-probability-details');
  await details.waitFor({ state: 'visible', timeout: 60_000 });
  if (await details.getAttribute('open') === null) await details.locator('summary').click();
}
