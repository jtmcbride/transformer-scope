import { expect, type Page, type TestInfo } from '@playwright/test';
import reference from '../fixtures/parent-probabilities-reference.json' with {type:'json'};
import { downloadStory } from './chosen-token-pair-scan';

export async function parentProbabilityComparisonWorkflow(page:Page,info?:TestInfo){
  let inferences=0;page.on('console',message=>{if(message.text()==='position-prefix')inferences++;});
  const shared=page.getByRole('region',{name:'Shared story continuations',exact:true});await shared.locator('.story-parent-comparison > summary').click();
  const panel=shared.getByRole('region',{name:'Parent versus new branch probabilities',exact:true}),before=await downloadStory(page,'Download workspace');
  await expect(panel.getByLabel('Parent comparison chosen token')).toHaveValue('471');
  await panel.getByRole('button',{name:'Measure parent and branch probabilities',exact:true}).click();
  await expect(shared.getByRole('button',{name:'Branch both stories with this pair',exact:true})).toHaveCount(0);
  await expect(shared.getByRole('button',{name:'Measure token-by-token interactions',exact:true})).toBeDisabled();
  await expect(page.getByRole('button',{name:'Scan both tokens',exact:true})).toBeDisabled();
  await expect(panel.getByTestId('parent-trace-status')).toHaveText('Parent comparison complete. 16 / 16 predictions measured.',{timeout:60000});
  const expected=reference.cases.find(row=>row.referenceIndex===6&&row.childBranch==='edited'&&row.targetTokenId===471)!,record=await downloadStory(page,'Download parent comparison');
  expect(record.format).toBe('transformer-story-parent-probabilities');expect(record.backend).toBe('cpu');expect(record.complete).toBe(true);expect(record.targetTokenId).toBe(471);
  for(const side of [0,1])for(const role of ['parent','child'])record.rows[side][role].forEach((point:any,i:number)=>{const r=(expected.rows[side] as any)[role][i];expect(point.source.prefix).toEqual(r.prefix);expect([point.offset,point.source.step,point.source.choice.tokenId]).toEqual([r.offset,r.step,r.tokenId]);expect(point.probability).toBeCloseTo(r.probability,5);});
  record.differences.forEach((track:any[],side:number)=>track.forEach(row=>{expect(row.difference).toBe(row.child.probability-row.parent.probability);expect(row.parent.source.step).toBe(row.child.source.step);expect(row.difference).toBeCloseTo(expected.rows[side].child[row.offset].probability-expected.rows[side].parent[row.offset].probability,5);}));
  const chart=panel.getByRole('region',{name:'Parent and new branch probability tracks',exact:true});expect(Number(await chart.getAttribute('data-probability-limit'))).toBe(Math.max(...record.rows.flatMap((row:any)=>[...row.parent,...row.child].map((point:any)=>point.probability))));
  const callsBeforeDisplay=inferences;
  await panel.getByLabel('Parent comparison display').selectOption('difference');expect(Number(await chart.getAttribute('data-difference-limit'))).toBe(Math.max(...record.differences.flat().map((row:any)=>Math.abs(row.difference))));
  const point=panel.locator('.parent-trace-point[data-side="0"][data-offset="1"]');await point.focus();await page.keyboard.press('Enter');await expect(panel.getByRole('button',{pressed:true})).toHaveCount(2);
  await page.keyboard.press('ArrowDown');await expect(panel.locator('.parent-trace-point[data-side="1"][data-offset="1"]')).toBeFocused();
  for(const side of [0,1])expect(parseFloat(await panel.getByTestId(`parent-trace-difference-${side}`).innerText())/100).toBeCloseTo(record.differences[side][1].difference,6);
  await expect(panel.getByRole('article',{name:'Story A parent comparison details',exact:true})).toContainText('Recorded choice reproduced.');
  expect((await downloadStory(page,'Download parent comparison')).rows).toEqual(record.rows);expect(inferences).toBe(callsBeforeDisplay);
  await page.setViewportSize({width:390,height:844});expect(await panel.evaluate(el=>el.scrollWidth<=el.clientWidth+1&&el.getBoundingClientRect().right<=innerWidth)).toBe(true);expect(await chart.evaluate(el=>el.scrollWidth>el.clientWidth)).toBe(true);
  for(const control of [point,panel.getByLabel('Parent comparison chosen token'),panel.getByRole('button',{name:'Measure parent and branch probabilities',exact:true})])expect((await control.boundingBox())!.height).toBeGreaterThanOrEqual(44);
  if(info)await panel.screenshot({path:info.outputPath('parent-versus-rebranch-probabilities-mobile.png')});
  await panel.getByLabel('Parent comparison new paths').selectOption('original');await expect(panel.locator('.parent-trace-point')).toHaveCount(0);await panel.getByRole('button',{name:'Measure parent and branch probabilities',exact:true}).click();await expect(panel.getByTestId('parent-trace-status')).toContainText('Parent comparison complete.',{timeout:60000});
  const unchanged=await downloadStory(page,'Download parent comparison');expect(unchanged.differences.flat().every((row:any)=>row.samePrefix&&row.difference===0)).toBe(true);
  expect((await downloadStory(page,'Download workspace')).nodes).toEqual(before.nodes);
  await shared.locator('.story-parent-comparison > summary').click();await shared.locator('.story-parent-comparison > summary').click();await expect(panel.locator('.parent-trace-point')).toHaveCount(0);await expect(panel.getByLabel('Parent comparison chosen token')).toHaveValue('471');await expect(page.getByRole('alert')).toHaveCount(0);
}
