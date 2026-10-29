import { expect, test } from '@playwright/test';
import input from '../fixtures/story-trajectory-workspace.json' with {type:'json'};
import fixed from '../fixtures/story-fixed-token-trajectories-reference.json' with {type:'json'};
import { downloadStory } from '../support/chosen-token-pair-scan';

test('saved chosen-token effects use aligned signed scales and effect trends, preserve path choices, and distinguish expiry',async({page},info)=>{
  test.setTimeout(180000);let calls=0;page.on('console',m=>{if(m.text()==='effect-prefix')calls++;});
  await page.route('**/model.worker-*.js',async route=>{const response=await route.fetch();await route.fulfill({response,body:`${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='BASELINE')console.info('effect-prefix');dispatch.call(self,e);};}`});});
  const workspace=structuredClone(input.workspace) as typeof input.workspace & {sharedForks:({targetTokenId?:number}&typeof input.workspace.sharedForks[number])[]};
  workspace.sharedForks[0].targetTokenId=471;workspace.sharedForks[1].targetTokenId=1;
  await page.goto('?backend=cpu');await expect(page.locator('.status')).toHaveText('Showing 1.00×',{timeout:60000});
  await page.getByLabel('Open story workspace').setInputFiles({name:'chosen-token-tracks.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(workspace))});
  await page.locator('.story-map-comparison > summary').click();
  const shared=page.getByRole('region',{name:'Shared story continuations',exact:true});
  const trace=shared.getByRole('region',{name:'Head-pair interaction along the stories',exact:true}),chart=trace.getByRole('region',{name:'Scrollable token interaction tracks',exact:true});
  await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-0');
  async function measure(targetIndex:number,caseIndex:number,branch:'original'|'edited'){
    const target=fixed.targets[targetIndex];await expect(trace.getByLabel('Chosen next token')).toHaveValue(String(target.tokenId));await expect(trace.getByLabel('Interaction trajectory display')).toHaveValue('effect');
    await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();
    await expect(trace.getByTestId('trajectory-status')).toHaveText('Token interactions complete. 12 / 12 tokens measured.',{timeout:60000});
    const record=await downloadStory(page,'Download token interactions');expect(record.display).toBe('effect');expect(record.effectDefinition).toContain('p(pair) - p(source settings)');expect(record.complete).toBe(true);
    for(const side of [0,1])record.rows[side].forEach((row:any,i:number)=>{const r=target.cases[caseIndex][branch][side][i];expect([row.tokenId,row.targetTokenId,row.active]).toEqual([r.tokenId,target.tokenId,r.active]);for(const field of ['before','after','delta'] as const)expect(row[field]).toBeCloseTo(r[field],5);});
    expect(Number(await chart.getAttribute('data-effect-limit'))).toBe(Math.max(...record.rows.flat().map((row:any)=>Math.abs(row.delta))));
    const values=await trace.locator('.trajectory-point').evaluateAll(items=>items.map(el=>Number(el.getAttribute('data-delta'))));expect(values).toEqual(record.rows.flat().map((row:any)=>row.delta));
    return record;
  }
  const first=await measure(0,0,'edited');
  // These probability-effect signs and trends differ from the interaction curve on the same points.
  await expect(trace.locator('.trajectory-point[data-track="0"][data-offset="1"]')).toContainText('Reversed');
  await trace.locator('.trajectory-point[data-track="0"][data-offset="2"]').focus();await page.keyboard.press('Enter');await expect(trace.getByRole('button',{pressed:true})).toHaveCount(2);
  for(const side of [0,1])expect(parseFloat(await trace.getByTestId(`trajectory-effect-${side}`).innerText())/100).toBeCloseTo(first.rows[side][2].delta,6);
  const measured=calls;for(const metric of ['probability','interaction','effect'])await trace.getByLabel('Interaction trajectory display').selectOption(metric);expect(calls).toBe(measured);expect((await downloadStory(page,'Download token interactions')).rows).toEqual(first.rows);
  await page.setViewportSize({width:390,height:844});expect(await trace.evaluate(el=>el.scrollWidth<=el.clientWidth+1&&el.getBoundingClientRect().right<=innerWidth)).toBe(true);expect(await chart.evaluate(el=>el.scrollWidth>el.clientWidth)).toBe(true);
  await trace.screenshot({path:info.outputPath('signed-probability-effects-mobile.png')});await page.setViewportSize({width:1440,height:1100});
  await trace.getByLabel('Interaction trajectory paths').selectOption('original');await expect(trace.locator('.trajectory-point')).toHaveCount(0);await measure(0,0,'original');
  await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-1');await expect(trace.locator('.trajectory-point')).toHaveCount(0);const expired=await measure(1,1,'edited');
  expect(expired.rows.flat().filter((row:any)=>row.index>0).every((row:any)=>!row.active&&row.delta===0)).toBe(true);await expect(trace.locator('.trajectory-point[data-active="false"]')).toHaveCount(10);await expect(trace.locator('.trajectory-point[data-active="false"]').first()).toContainText('Expired');
  const after=await downloadStory(page,'Download workspace');expect(after.nodes).toEqual(workspace.nodes);expect(after.sharedForks).toEqual(workspace.sharedForks);
  await page.reload();await page.locator('.story-map-comparison > summary').click();await expect(trace.getByLabel('Chosen next token')).toHaveValue('1');await expect(trace.getByLabel('Interaction trajectory display')).toHaveValue('effect');await expect(trace.locator('.trajectory-point')).toHaveCount(0);
  // Import can replace only the objective while keeping identical story/group IDs and frames.
  const replacement=structuredClone(workspace);replacement.sharedForks[1].targetTokenId=471;
  async function replace(){await page.getByLabel('Open story workspace').setInputFiles({name:'replaced-objective.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(replacement))});await expect(shared).toBeHidden();await page.locator('.story-map-comparison > summary').click();}
  await replace();await expect(trace.getByLabel('Chosen next token')).toHaveValue('471');await expect(trace.getByLabel('Interaction trajectory display')).toHaveValue('effect');
  delete replacement.sharedForks[1].targetTokenId;await replace();await expect(trace.getByLabel('Interaction trajectory target mode')).toHaveValue('generated');await expect(trace.getByLabel('Interaction trajectory display')).toHaveValue('interaction');
  await expect(page.getByRole('alert')).toHaveCount(0);
});
