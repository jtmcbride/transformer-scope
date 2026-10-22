import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import reference from '../fixtures/story-head-scan-reference.json' with { type: 'json' };
import forks from '../fixtures/story-forks-reference.json' with { type: 'json' };
import probabilities from '../fixtures/branch-probabilities-reference.json' with { type: 'json' };
async function ready(page: Page, edited=false) {
  await page.goto('?backend=cpu'); await expect(page.locator('.status')).toHaveText('Showing 1.00×',{timeout:60000});
  await page.getByLabel('Story prompt').fill('Tom found a red ball. He wanted to');
  await page.getByRole('button',{name:'Run input',exact:true}).click();
  await expect(page.locator('.visual-answer strong')).toHaveText('play');
  if(edited) { await page.getByRole('button',{name:'Mute selected head',exact:true}).click(); await expect(page.locator('.status')).toHaveText('Showing 0.00×'); await page.getByLabel('Compare with current head edit').check(); }
  await page.getByLabel('New-token limit').fill('8'); await page.getByRole('button',{name:'Play story',exact:true}).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.',{timeout:60000});
  await page.getByRole('button',{name:edited ? /^Inspect edited token 7:/ : /^Inspect original token 6:/}).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({timeout:60000});
}
test('scan ranks all changes, fills the chosen draft and preview, and branches with the measured edit', async ({page},info)=>{
  test.setTimeout(180000); await ready(page);
  const scan=page.getByRole('region',{name:'Scan heads at selected token',exact:true});
  await scan.getByRole('button',{name:'Scan all heads',exact:true}).click();
  await expect(scan.getByTestId('head-scan-status')).toHaveText('Scan complete. 48 / 48 changes measured.',{timeout:90000});
  const expected=[...reference.cases[0].rows].sort((a,b)=>Math.abs(b.after-reference.cases[0].source.choice.probability)-Math.abs(a.after-reference.cases[0].source.choice.probability)||a.edit.layer-b.edit.layer||a.edit.head-b.edit.head||a.edit.factor-b.edit.factor);
  const list=scan.getByRole('list',{name:'Ranked head changes'});
  for(let i=0;i<8;i++) { const row=list.locator('li').nth(i); await expect(row).toHaveAttribute('data-scan-layer',String(expected[i].edit.layer)); await expect(row).toHaveAttribute('data-scan-head',String(expected[i].edit.head)); await expect(row).toHaveAttribute('data-scan-factor',String(expected[i].edit.factor)); }
  await scan.getByRole('button',{name:'Show all 48 results',exact:true}).click();
  for(const row of reference.cases[0].rows) {
    const measured=scan.locator(`li[data-scan-layer="${row.edit.layer}"][data-scan-head="${row.edit.head}"][data-scan-factor="${row.edit.factor}"]`);
    expect(parseFloat(await measured.getByTestId('scan-after').innerText())/100).toBeCloseTo(row.after,5);
    expect(parseFloat(await measured.getByTestId('scan-delta').innerText())/100).toBeCloseTo(row.after-reference.cases[0].source.choice.probability,5);
  }
  for(const order of ['increase','decrease']) { await scan.getByLabel('Sort head scan results').selectOption(order); const values=await list.getByTestId('scan-delta').allTextContents(); const nums=values.map(parseFloat); expect(nums.every((n,i)=>!i||(order==='increase'?n<=nums[i-1]:n>=nums[i-1]))).toBe(true); }
  const first=expected[0].edit; await scan.getByRole('button',{name:`Use layer ${first.layer+1} head ${first.head+1} strength ${first.factor}`,exact:true}).click();
  const inspection=page.getByRole('region',{name:'Generated token computation',exact:true});
  await expect(inspection.getByRole('button',{name:`L${first.layer+1} · Head ${first.head+1}`,exact:true})).toHaveAttribute('aria-pressed','true');
  await expect(page.getByLabel('Branch head strength')).toHaveValue(String(first.factor));
  await page.setViewportSize({width:390,height:844}); expect(await scan.evaluate(el=>el.scrollWidth<=el.clientWidth+1&&el.getBoundingClientRect().right<=innerWidth)).toBe(true);
  await page.setViewportSize({width:1440,height:1100}); await scan.getByRole('button',{name:'Show strongest eight',exact:true}).click(); await scan.screenshot({path:info.outputPath('head-scan.png')}); await scan.getByRole('button',{name:'Show all 48 results',exact:true}).click();
  await scan.getByRole('button',{name:'Use layer 1 head 1 strength 0',exact:true}).focus(); await page.keyboard.press('Enter');
  const preview=page.getByRole('region',{name:'Scanned head probability preview',exact:true});
  await expect(preview.getByTestId('branch-choice-explanation')).toBeVisible({timeout:60000});
  const selected=preview.getByRole('list',{name:'Selected token probability'});
  expect(parseFloat(await selected.getByTestId('probability-after').innerText())/100).toBeCloseTo(reference.cases[0].rows[0].after,5);
  await page.getByLabel('Branch new-token limit').fill('3');
  await page.getByRole('button',{name:'Compare from this token',exact:true}).click();
  await expect(page.locator('.fork-status')).toHaveText('Branch comparison complete.',{timeout:60000});
  await expect(preview).toHaveCount(0);
  const wait=page.waitForEvent('download'); await page.getByRole('button',{name:'Download branch comparison',exact:true}).click();
  const saved=JSON.parse(await readFile((await(await wait).path())!,'utf8'));
  expect(saved.plan.edit).toMatchObject({layer:0,head:0,factor:0});
  expect(saved.frame.edited.choices.map((c:{tokenId:number})=>c.tokenId)).toEqual(forks.cases[0].edited.choices.slice(0,3).map(c=>c.tokenId));
});
test('scan cancellation and navigation discard late results, and scans retain an edited source',async({page})=>{
  test.setTimeout(180000);
  await page.route('**/model.worker-*.js',async route=>{const response=await route.fetch(); await route.fulfill({response,body:`${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='BASELINE'){console.info('scan-baseline');setTimeout(()=>dispatch.call(self,e),150);}else dispatch.call(self,e);};}`});});
  await ready(page,true);
  const scan=page.getByRole('region',{name:'Scan heads at selected token',exact:true});
  let pending=page.waitForEvent('console',m=>m.text()==='scan-baseline');
  await scan.getByRole('button',{name:'Scan all heads',exact:true}).click(); await pending;
  await scan.getByRole('button',{name:'Stop head scan',exact:true}).click(); await page.waitForTimeout(400);
  await expect(scan.getByTestId('head-scan-status')).toContainText('0 / 48'); await expect(scan.getByRole('list')).toHaveCount(0);
  await scan.getByRole('button',{name:'Scan all heads',exact:true}).click();
  await expect(scan.getByTestId('head-scan-status')).toContainText('Scan complete. 48 / 48',{timeout:90000});
  await scan.getByRole('button',{name:'Show all 48 results',exact:true}).click();
  const row=scan.locator('li[data-scan-layer="1"][data-scan-head="1"][data-scan-factor="0"]');
  const expected=probabilities.cases[2].rows.find(r=>r.tokenId===probabilities.cases[2].plan.source.choice.tokenId)!;
  expect(parseFloat(await row.getByTestId('scan-after').innerText())/100).toBeCloseTo(expected.after,5);
  await scan.getByRole('button',{name:'Use layer 2 head 2 strength 0',exact:true}).click();
  await page.getByLabel('Branch new-token limit').fill('1');
  await page.getByRole('button',{name:'Compare from this token',exact:true}).click();
  await expect(page.locator('.fork-status')).toHaveText('Branch comparison complete.',{timeout:60000});
  await expect(page.getByRole('region',{name:'Head change probabilities',exact:true}).getByTestId('branch-choice-explanation')).toBeVisible({timeout:60000});
  pending=page.waitForEvent('console',m=>m.text()==='scan-baseline'); await scan.getByRole('button',{name:'Scan all heads',exact:true}).click(); await pending;
  await page.getByRole('button',{name:/^Inspect keep source settings token 7:/}).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({timeout:60000});
  await page.getByRole('button',{name:'Back to previous comparison',exact:true}).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({timeout:60000});
  await expect(page.getByRole('button',{name:'Start new branch comparison',exact:true})).toBeEnabled();
  await expect(scan.getByTestId('head-scan-status')).toHaveText('Ready to scan. 0 / 48 changes measured.');

  pending=page.waitForEvent('console',m=>m.text()==='scan-baseline'); await scan.getByRole('button',{name:'Scan all heads',exact:true}).click(); await pending;
  await page.getByRole('button',{name:'Next token →',exact:true}).click();
  await expect(page.getByTestId('selected-token-result')).toBeVisible({timeout:60000}); await page.waitForTimeout(400);
  await expect(scan.getByTestId('head-scan-status')).toHaveText('Ready to scan. 0 / 48 changes measured.'); await expect(page.getByRole('alert')).toHaveCount(0);
});
