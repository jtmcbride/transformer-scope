import { expect, type Page, type TestInfo } from '@playwright/test';
import input from '../fixtures/story-trajectory-workspace.json' with {type:'json'};
import reference from '../fixtures/trajectory-rebranches-reference.json' with {type:'json'};
import type { StoryWorkspace } from '../../src/story-workspace';
import { downloadStory } from './chosen-token-pair-scan';

export async function openPositionRanking(page:Page){
  const workspace=structuredClone(input.workspace) as unknown as StoryWorkspace;
  workspace.sharedForks!.forEach(group=>group.targetTokenId=471);
  await page.goto('?backend=cpu');await expect(page.locator('.status')).toHaveText('Showing 1.00×',{timeout:60000});
  await page.getByLabel('Open story workspace').setInputFiles({name:'position-ranking.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(workspace))});
  await page.locator('.story-map-comparison > summary').click();
  const shared=page.getByRole('region',{name:'Shared story continuations',exact:true});await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-0');
  return {workspace,shared,trace:shared.getByRole('region',{name:'Head-pair interaction along the stories',exact:true})};
}
export async function trajectoryPositionBranchWorkflow(page:Page,info?:TestInfo){
  let calls=0;page.on('console',m=>{if(m.text()==='position-prefix')calls++;});
  await page.route('**/model.worker-*.js',async route=>{const response=await route.fetch();await route.fulfill({response,body:`${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='BASELINE')console.info('position-prefix');dispatch.call(self,e);};}`});});
  const {workspace,shared,trace}=await openPositionRanking(page);
  await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();await expect(trace.getByTestId('trajectory-status')).toHaveText('Token interactions complete. 12 / 12 tokens measured.',{timeout:60000});
  const record=await downloadStory(page,'Download token interactions'),measured=calls;
  const ranking=trace.getByRole('region',{name:'Ranked probability-effect differences',exact:true}),list=ranking.getByRole('list',{name:'Token positions ranked by probability-effect difference',exact:true});
  await expect(ranking.getByTestId('trajectory-ranking-status')).toContainText('6 / 6 offsets measured in both stories. Complete');
  for(const order of ['absolute','increase','decrease']){
    await ranking.getByLabel('Rank token-position differences').selectOption(order);
    const expected=record.rows[0].map((a:any,i:number)=>({index:a.index,difference:record.rows[1][i].delta-a.delta})).sort((a:any,b:any)=>(order==='absolute'?Math.abs(b.difference)-Math.abs(a.difference):order==='increase'?b.difference-a.difference:a.difference-b.difference)||a.index-b.index);
    const actual=await list.locator('li').evaluateAll(items=>items.map(el=>({index:Number(el.getAttribute('data-offset')),difference:Number(el.getAttribute('data-difference'))})));expect(actual).toEqual(expected);
    const exported=await downloadStory(page,'Download token interactions');expect(exported.positionDifferenceOrder).toBe(order);expect(exported.positionDifferences.map((row:any)=>({index:row.index,difference:row.difference}))).toEqual(expected);
  }
  await ranking.getByLabel('Rank token-position differences').selectOption('absolute');
  const expected=reference.cases.find(row=>row.groupId==='trajectory-group-0'&&row.branch==='edited'&&row.strengths==='normal'&&row.duration==='continuation')!;
  const jump=list.locator('li').first().getByRole('button');await jump.focus();await page.keyboard.press('Enter');
  await expect(trace.locator(`.trajectory-point[data-track="0"][data-offset="${expected.index}"]`)).toBeFocused();await expect(trace.getByRole('button',{pressed:true})).toHaveCount(2);
  await expect(trace.getByRole('article',{name:'Story A selected interaction',exact:true})).toContainText(`story step ${expected.plans[0].source.step}`);
  await ranking.getByRole('button',{name:`Set up both branches from offset ${expected.index+1}`,exact:true}).click();
  const setup=shared.getByRole('group',{name:'Rebranch both stories from trajectory',exact:true});await expect(setup).toContainText('already active');await expect(setup.getByLabel('Rebranch first head strength')).toBeFocused();
  await list.locator('li').nth(1).getByRole('button').click();await expect(setup).toHaveCount(0);
  await jump.click();await ranking.getByRole('button',{name:`Set up both branches from offset ${expected.index+1}`,exact:true}).click();
  await setup.getByLabel('Rebranch first head strength').selectOption('1');await setup.getByLabel('Rebranch second head strength').selectOption('1');
  await shared.getByLabel('Shared continuation token limit').fill('4');expect(calls).toBe(measured);
  await page.setViewportSize({width:390,height:844});expect(await ranking.evaluate(el=>el.scrollWidth<=el.clientWidth+1&&el.getBoundingClientRect().right<=innerWidth)).toBe(true);
  for(const control of [jump,ranking.getByRole('button',{name:`Set up both branches from offset ${expected.index+1}`,exact:true}),setup.getByLabel('Rebranch first head strength'),setup.getByRole('button',{name:'Cancel trajectory branch setup',exact:true})]){const size=(await control.boundingBox())!;expect(size.height).toBeGreaterThanOrEqual(44);}
  if(info){await ranking.screenshot({path:info.outputPath('ranked-token-effect-differences-mobile.png')});await setup.screenshot({path:info.outputPath('ranked-point-shared-branch-setup-mobile.png')});}
  await shared.getByRole('button',{name:'Branch both stories with this pair',exact:true}).click();await expect(shared.getByTestId('shared-continuation-status')).toHaveText('Shared continuations complete.',{timeout:60000});
  const child=await downloadStory(page,'Download shared comparison');expect(child.group.targetTokenId).toBe(471);
  child.nodes.forEach((node:any,side:number)=>{expect(node.parentId).toBe(expected.nodeIds[side]);expect(node.plan.source).toEqual(expected.plans[side].source);expect([node.plan.edit,...node.plan.additionalEdits]).toEqual([expected.plans[side].edit,...expected.plans[side].additionalEdits]);
    for(const branch of ['original','edited'] as const){expect(node.frame[branch].tokenIds).toEqual(expected.frames[side][branch].tokenIds);node.frame[branch].choices.forEach((choice:any,i:number)=>expect(choice.probability).toBeCloseTo(expected.frames[side][branch].choices[i].probability,5));}
  });
  const saved=await downloadStory(page,'Download workspace');expect(saved.root).toEqual(workspace.root);expect(saved.nodes.slice(0,workspace.nodes.length)).toEqual(workspace.nodes);expect(saved.nodes).toHaveLength(workspace.nodes.length+2);expect(saved.sharedForks).toHaveLength(3);
  await page.reload();await page.locator('.story-map-comparison > summary').click();expect((await downloadStory(page,'Download shared comparison')).nodes).toEqual(child.nodes);await expect(trace.getByLabel('Chosen next token')).toHaveValue('471');await expect(trace.locator('.trajectory-point')).toHaveCount(0);
  await page.getByLabel('Open story workspace').setInputFiles({name:'nested-workspace.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(saved))});
  await expect(shared).toBeHidden();await page.locator('.story-map-comparison > summary').click();
  await expect(shared.getByTestId('shared-continuation-status')).toHaveText('Shared continuations complete.');
  await expect(page.getByRole('alert')).toHaveCount(0);
  return {child,saved};
}
