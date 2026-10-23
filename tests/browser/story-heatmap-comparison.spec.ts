import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import reference from '../fixtures/story-heatmap-comparison-reference.json' with { type:'json' };
const key = (edits:{layer:number;head:number;factor:number}[]) => edits.map(e=>`${e.layer}:${e.head}:${e.factor}`).join('|');
async function download(page:Page,name:string) {
  const waiting=page.waitForEvent('download');await page.getByRole('button',{name,exact:true}).click();
  return JSON.parse(await readFile((await (await waiting).path())!,'utf8'));
}
async function ready(page:Page) {
  await page.goto('?backend=cpu');await expect(page.locator('.status')).toHaveText('Showing 1.00×',{timeout:60000});
  await page.getByLabel('Story prompt').fill('Tom found a red ball. He wanted to');await page.getByRole('button',{name:'Run input',exact:true}).click();
  await expect(page.locator('.visual-answer strong')).toHaveText('play');
  await page.getByLabel('New-token limit').fill('8');await page.getByRole('button',{name:'Play story',exact:true}).click();
  await expect(page.locator('.continuation-status')).toHaveText('Continuation complete.',{timeout:60000});
  await page.getByRole('button',{name:/^Inspect original token 6:/}).click();await expect(page.getByTestId('selected-token-result')).toBeVisible({timeout:60000});
  await page.getByLabel('Branch new-token limit').fill('6');await page.getByRole('button',{name:'Compare from this token',exact:true}).click();
  await expect(page.locator('.fork-status')).toHaveText('Branch comparison complete.',{timeout:60000});
  await expect(page.getByRole('region',{name:'Head change probabilities',exact:true}).getByTestId('probability-after').first()).toBeVisible({timeout:60000});
  await page.locator('.story-map-comparison > summary').click();
  const panel=page.getByRole('region',{name:'Compare story interaction heatmaps',exact:true});
  await expect(panel.getByLabel('Story A path').locator('option:checked')).toHaveText('Comparison 1 · Keep source settings');
  await panel.getByLabel('Token A position').selectOption('1');await panel.getByLabel('Token B position').selectOption('1');
  return panel;
}
test('compares real branched-token maps and differences against PyTorch, with synchronized selection and inference-free swapping',async({page},info)=>{
  test.setTimeout(300000);const panel=await ready(page),before=await download(page,'Download workspace');
  await panel.getByRole('button',{name:'Scan both tokens',exact:true}).click();
  await expect(page.getByRole('button',{name:'Scan head pairs',exact:true})).toBeDisabled();
  await expect(page.getByRole('button',{name:'Start new branch comparison',exact:true})).toBeDisabled();
  await expect(panel.getByTestId('comparison-map-status')).toHaveText('Heatmap comparison complete. A: 120 / 120 · B: 120 / 120 · 120 matching combinations.',{timeout:240000});
  const saved=await download(page,'Download heatmap comparison');expect(saved.complete).toBe(true);expect(saved.checkpoint).toContain(reference.modelRevision);
  for (const side of [0,1]) {
    expect(saved.sources[side].step).toBe(7);expect(saved.sources[side].prefix).toEqual(reference.cases[side].source.prefix);
    expect(saved.sources[side].choice.tokenId).toBe(reference.cases[side].source.choice.tokenId);expect(saved.sources[side].edits).toEqual(reference.cases[side].source.edits);
    expect(saved.sources[side].choice.probability).toBeCloseTo(reference.cases[side].source.choice.probability,5);
    expect(saved.rows[side]).toHaveLength(120);
    saved.rows[side].forEach((row:any,i:number)=>{
      const expected=reference.cases[side].rows[i];expect(row.edits).toEqual(expected.edits);expect(row.after).toBeCloseTo(expected.after,5);
      row.individualProbabilities.forEach((p:number,j:number)=>expect(p).toBeCloseTo(expected.individualProbabilities[j],5));
      expect(Math.abs(row.interaction-expected.interaction)).toBeLessThanOrEqual(4*5e-6);
    });
  }
  const a=panel.getByRole('article',{name:'Token A interaction map',exact:true}), b=panel.getByRole('article',{name:'Token B interaction map',exact:true}), difference=panel.getByRole('article',{name:'Interaction difference map',exact:true});
  for(const map of [a,b,difference])await expect(map.locator('.pair-map-cell')).toHaveCount(240);
  expect(await a.getAttribute('data-color-limit')).toBe(await b.getAttribute('data-color-limit'));
  expect(Number(await a.getAttribute('data-color-limit'))).toBe(Math.max(...saved.rows.flat().map((row:any)=>Math.abs(row.interaction))));
  const cells=await difference.locator('.pair-map-cell').evaluateAll(nodes=>nodes.map(n=>({key:n.getAttribute('data-pair-key'),value:Number(n.getAttribute('data-interaction'))})));
  for(const cell of cells){const expected=reference.differences.find(row=>key(row.edits)===cell.key)!;expect(Math.abs(cell.value-expected.difference)).toBeLessThanOrEqual(8*5e-6);}
  const strongest=[...reference.differences].sort((x,y)=>Math.abs(y.difference)-Math.abs(x.difference))[0],selectedKey=key(strongest.edits);
  await expect(panel.getByRole('list',{name:'Largest interaction changes'}).locator('li').first()).toHaveAttribute('data-pair-key',selectedKey);
  const cell=difference.locator(`button[data-pair-key="${selectedKey}"]`).first();await cell.focus();await page.keyboard.press('Enter');
  for(const map of [a,b,difference])await expect(map.getByRole('button',{pressed:true})).toHaveCount(2);
  expect(Math.abs(parseFloat(await panel.getByTestId('compared-interaction-difference').innerText())/100-strongest.difference)).toBeLessThanOrEqual(8*5e-6+0.5e-6);
  await panel.screenshot({path:info.outputPath('compared-story-maps.png')});
  await page.setViewportSize({width:390,height:844});expect(await panel.evaluate(el=>el.scrollWidth<=el.clientWidth+1&&el.getBoundingClientRect().right<=innerWidth)).toBe(true);
  for(const map of [a,b,difference])expect(await map.locator('.pair-map-scroll').evaluate(el=>el.scrollWidth>el.clientWidth)).toBe(true);
  if(info.project.use.hasTouch){expect(await page.evaluate(()=>matchMedia('(pointer:coarse)').matches)).toBe(true);const size=await difference.locator('.pair-map-cell').first().boundingBox();expect(size!.width).toBeGreaterThanOrEqual(44);expect(size!.height).toBeGreaterThanOrEqual(44);}
  await page.setViewportSize({width:1440,height:1100});
  await panel.getByRole('button',{name:'Swap A and B',exact:true}).click();
  const swapped=await download(page,'Download heatmap comparison');expect(swapped.rows).toEqual([saved.rows[1],saved.rows[0]]);
  swapped.changes.forEach((r:any,i:number)=>expect(r.difference).toBeCloseTo(-saved.changes[i].difference,14));
  expect(Number(await difference.locator(`button[data-pair-key="${selectedKey}"]`).first().getAttribute('data-interaction'))).toBe(-saved.changes.find((r:any)=>r.key===selectedKey).difference);
  for(const order of ['increase','decrease']){await panel.getByLabel('Rank interaction differences').selectOption(order);const expected=[...swapped.changes].sort((x:any,y:any)=>order==='increase'?y.difference-x.difference:x.difference-y.difference)[0];await expect(panel.getByRole('list',{name:'Largest interaction changes'}).locator('li').first()).toHaveAttribute('data-pair-key',expected.key);}
  await panel.getByRole('button',{name:'Open story A',exact:true}).click();await expect(page.getByTestId('selected-token-result')).toBeVisible({timeout:60000});
  await expect(panel.getByTestId('comparison-map-status')).toContainText('120 matching combinations');
  const after=await download(page,'Download workspace');expect(after.root).toEqual(before.root);expect(after.nodes).toEqual(before.nodes);
  await expect(page.getByRole('alert')).toHaveCount(0);
});
test('partial comparisons keep unmatched cells missing, cancel on source changes, and exclude competing scans',async({page})=>{
  test.setTimeout(240000);
  await page.route('**/model.worker-*.js',async route=>{const response=await route.fetch();await route.fulfill({response,body:`${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='BASELINE'){console.info('comparison-baseline');setTimeout(()=>dispatch.call(self,e),600);}else dispatch.call(self,e);};}`});});
  const panel=await ready(page);
  await panel.getByRole('button',{name:'Scan both tokens',exact:true}).click();
  await expect(panel.getByTestId('comparison-map-status')).toContainText('A: 120 / 120 · B: 0 / 120',{timeout:180000});
  await panel.getByRole('button',{name:'Stop map comparison',exact:true}).click();
  await expect(panel.getByTestId('comparison-map-status')).toContainText('Comparison stopped.');
  const difference=panel.getByRole('article',{name:'Interaction difference map',exact:true});await expect(difference.locator('.pair-map-cell')).toHaveCount(0);await expect(difference.locator('.pair-map-pending')).toHaveCount(240);
  await panel.getByRole('article',{name:'Token A interaction map',exact:true}).locator('.pair-map-cell').first().click();await expect(panel.getByTestId('compared-interaction-difference')).toHaveText('Not measured');
  let pending=page.waitForEvent('console',m=>m.text()==='comparison-baseline');await panel.getByRole('button',{name:'Scan both tokens',exact:true}).click();await pending;
  await panel.getByLabel('Token B position').selectOption('2');
  await expect(panel.getByTestId('comparison-map-status')).toHaveText('Ready to compare maps. A: 0 / 120 · B: 0 / 120 · 0 matching combinations.');
  await expect(page.getByRole('button',{name:'Scan head pairs',exact:true})).toBeEnabled();
  await panel.getByLabel('Comparison scan strengths').selectOption('all');await expect(panel.getByLabel('Comparison heatmap strength assignment').locator('option')).toHaveCount(9);
  await panel.getByLabel('Comparison heatmap strength assignment').selectOption('2:1');
  pending=page.waitForEvent('console',m=>m.text()==='comparison-baseline');await panel.getByRole('button',{name:'Scan both tokens',exact:true}).click();await pending;
  await page.locator('.story-map-comparison > summary').click();await expect(page.getByRole('button',{name:'Scan head pairs',exact:true})).toBeEnabled();
  await page.locator('.story-map-comparison > summary').click();await expect(panel.getByTestId('comparison-map-status')).toContainText('Comparison stopped.');
  const pairScan=page.getByRole('region',{name:'Scan head pairs at selected token',exact:true});
  await pairScan.getByRole('button',{name:'Scan head pairs',exact:true}).click();await expect(panel.getByRole('button',{name:'Scan both tokens',exact:true})).toBeDisabled();
  await pairScan.getByRole('button',{name:'Stop pair scan',exact:true}).click();await expect(panel.getByRole('button',{name:'Scan both tokens',exact:true})).toBeEnabled();
  await page.getByRole('button',{name:'Start new branch comparison',exact:true}).click();
  await panel.getByRole('button',{name:'Scan both tokens',exact:true}).click();
  await expect(page.locator('.fork-status')).toContainText('Comparison paused after');
  await panel.getByRole('button',{name:'Stop map comparison',exact:true}).click();
  await page.getByRole('button',{name:'Reset story',exact:true}).click();await expect(page.locator('.story-map-comparison')).toHaveCount(0);await expect(page.getByRole('alert')).toHaveCount(0);
});
