import { expect, type Page, type TestInfo } from '@playwright/test';
import reference from '../fixtures/parent-position-rebranches-reference.json' with {type:'json'};
import { downloadStory } from './chosen-token-pair-scan';

/** Start with the CPU-generated children produced by trajectoryPositionBranchWorkflow. */
export async function parentPositionBranchWorkflow(page:Page,info?:TestInfo){
  let inferences=0;page.on('console',message=>{if(message.text()==='position-prefix')inferences++;});
  const shared=page.getByRole('region',{name:'Shared story continuations',exact:true}),before=await downloadStory(page,'Download workspace'),source=await downloadStory(page,'Download shared comparison');
  await shared.locator('.story-parent-comparison > summary').click();const panel=shared.getByRole('region',{name:'Parent versus new branch probabilities',exact:true});
  await expect(panel.getByLabel('Parent comparison chosen token')).toHaveValue('471');await panel.getByRole('button',{name:'Measure parent and branch probabilities',exact:true}).click();
  await expect(panel.getByTestId('parent-trace-status')).toHaveText('Parent comparison complete. 16 / 16 predictions measured.',{timeout:60000});
  const record=await downloadStory(page,'Download parent comparison'),measured=inferences;
  const ranking=panel.getByRole('region',{name:'Ranked parent-to-new probability changes',exact:true}),list=ranking.getByRole('list',{name:'Positions ranked by parent-to-new probability change',exact:true});
  await expect(ranking.getByTestId('parent-ranking-status')).toContainText('4 / 4 offsets measured on all four paths. Complete');
  for(const order of ['absolute','increase','decrease']){
    await ranking.getByLabel('Rank parent-to-new probability changes').selectOption(order);
    const score=(a:number,b:number)=>order==='absolute'?Math.max(Math.abs(a),Math.abs(b)):order==='increase'?Math.max(a,b):Math.min(a,b);
    const expected=record.differences[0].map((a:any,i:number)=>({offset:a.offset,a:a.difference,b:record.differences[1][i].difference,score:score(a.difference,record.differences[1][i].difference)})).sort((a:any,b:any)=>(order==='decrease'?a.score-b.score:b.score-a.score)||a.offset-b.offset);
    const actual=await list.locator('li').evaluateAll(items=>items.map(el=>({offset:Number(el.getAttribute('data-offset')),a:Number(el.getAttribute('data-a-change')),b:Number(el.getAttribute('data-b-change')),score:Number(el.getAttribute('data-score'))})));expect(actual).toEqual(expected);
    const independent=reference.cases.find(row=>row.referenceIndex===6&&row.branch==='edited'&&row.targetTokenId===471&&row.order===order)!;expect(actual.map(row=>row.offset)).toEqual(independent.ranked.map(row=>row.offset));
    const exported=await downloadStory(page,'Download parent comparison');expect(exported.positionChangeOrder).toBe(order);expect(exported.positionChanges.map((row:any)=>row.offset)).toEqual(expected.map((row:any)=>row.offset));expect(exported.rows).toEqual(record.rows);
  }
  await ranking.getByLabel('Rank parent-to-new probability changes').selectOption('absolute');
  const expected=reference.cases.find(row=>row.referenceIndex===6&&row.branch==='edited'&&row.targetTokenId===471&&row.order==='absolute'&&row.strengths==='amplify'&&row.duration==='continuation')!;
  const jump=list.locator('li').first().getByRole('button');await jump.focus();await page.keyboard.press('Enter');
  await expect(panel.locator(`.parent-trace-point[data-side="0"][data-offset="${expected.offset}"]`)).toBeFocused();await expect(panel.getByRole('button',{pressed:true})).toHaveCount(2);await expect(panel.getByLabel('Parent comparison display')).toHaveValue('difference');
  for(const side of [0,1])await expect(panel.getByRole('article',{name:`Story ${side===0?'A':'B'} parent comparison details`,exact:true})).toContainText(`step ${expected.plans[side].source.step}`);
  const prepare=ranking.getByRole('button',{name:`Set up both new branches from offset ${expected.offset+1}`,exact:true});await prepare.click();
  const setup=shared.getByRole('group',{name:'Rebranch both stories from parent comparison',exact:true});await expect(setup.getByLabel('Rebranch first head strength')).toBeFocused();await expect(setup).toContainText('already active');
  await list.locator('li').nth(1).getByRole('button').click();await expect(setup).toHaveCount(0);await jump.click();await prepare.click();
  await setup.getByLabel('Rebranch first head strength').selectOption('2');await setup.getByLabel('Rebranch second head strength').selectOption('2');await shared.getByLabel('Shared head-change duration').selectOption('continuation');await shared.getByLabel('Shared continuation token limit').fill('3');
  expect(inferences).toBe(measured);expect((await downloadStory(page,'Download workspace')).nodes).toEqual(before.nodes);
  await page.setViewportSize({width:390,height:844});expect(await ranking.evaluate(el=>el.scrollWidth<=el.clientWidth+1&&el.getBoundingClientRect().right<=innerWidth)).toBe(true);
  for(const control of [jump,prepare,ranking.getByLabel('Rank parent-to-new probability changes'),setup.getByLabel('Rebranch first head strength'),setup.getByRole('button',{name:'Cancel parent comparison branch setup',exact:true})])expect((await control.boundingBox())!.height).toBeGreaterThanOrEqual(44);
  if(info){await ranking.screenshot({path:info.outputPath('ranked-parent-to-new-probability-changes-mobile.png')});await setup.screenshot({path:info.outputPath('ranked-parent-position-rebranch-setup-mobile.png')});}
  await shared.getByRole('button',{name:'Branch both stories with this pair',exact:true}).click();await expect(shared.getByTestId('shared-continuation-status')).toHaveText('Shared continuations complete.',{timeout:60000});
  const child=await downloadStory(page,'Download shared comparison');expect(child.group.targetTokenId).toBe(471);
  child.nodes.forEach((node:any,side:number)=>{
    expect(node.parentId).toBe(source.nodes[side].id);expect(node.plan.source).toEqual(record.rows[side].child[expected.offset].source);expect(node.plan.source.prefix).toEqual(expected.plans[side].source.prefix);expect(node.plan.source.step).toBe(expected.plans[side].source.step);expect(node.plan.duration).toBe('continuation');expect([node.plan.edit,...node.plan.additionalEdits]).toEqual([expected.plans[side].edit,...expected.plans[side].additionalEdits]);
    for(const branch of ['original','edited'] as const){expect(node.frame[branch].tokenIds).toEqual(expected.frames[side][branch].tokenIds);node.frame[branch].choices.forEach((choice:any,i:number)=>expect(choice.probability).toBeCloseTo(expected.frames[side][branch].choices[i].probability,5));}
  });
  const saved=await downloadStory(page,'Download workspace');expect(saved.root).toEqual(before.root);expect(saved.nodes.slice(0,before.nodes.length)).toEqual(before.nodes);expect(saved.nodes).toHaveLength(before.nodes.length+2);expect(saved.sharedForks).toHaveLength(before.sharedForks.length+1);
  await page.reload();await page.locator('.story-map-comparison > summary').click();expect((await downloadStory(page,'Download shared comparison')).nodes).toEqual(child.nodes);await expect(shared.getByLabel('Chosen next token')).toHaveValue('471');
  await page.getByLabel('Open story workspace').setInputFiles({name:'ranked-parent-descendants.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved))});
  // Opening a separate workspace disposes the previous analysis and starts with it closed.
  await expect(shared).toBeHidden();await page.locator('.story-map-comparison > summary').click();
  expect((await downloadStory(page,'Download shared comparison')).nodes).toEqual(child.nodes);await expect(page.getByRole('alert')).toHaveCount(0);
}
