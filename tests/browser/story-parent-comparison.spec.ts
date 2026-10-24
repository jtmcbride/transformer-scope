import { expect, test } from '@playwright/test';
import input from '../fixtures/story-trajectory-workspace.json' with {type:'json'};
import { trajectoryPositionBranchWorkflow } from '../support/trajectory-position-branches';
import { parentProbabilityComparisonWorkflow } from '../support/parent-probability-comparison';
import { downloadStory } from '../support/chosen-token-pair-scan';

test('compares actual parents against new continuations for one chosen token, with shared scales, keyboard jumps and reference paths',async({page},info)=>{test.setTimeout(180000);await trajectoryPositionBranchWorkflow(page);await parentProbabilityComparisonWorkflow(page,info);});
test('parent comparisons preserve unavailable tails, keep partial results and cancel stale targets or closed views',async({page})=>{
  test.setTimeout(180000);
  const workspace=structuredClone(input.workspace);const parent=workspace.nodes[0];parent.frame.step=3;
  for(const branch of ['original','edited'] as const){parent.frame[branch].choices.splice(3);parent.frame[branch].tokenIds.splice(parent.plan.source.prefix.length+3);delete (parent.frame[branch] as {finish?:string}).finish;}
  await page.route('**/model.worker-*.js',async route=>{const response=await route.fetch();await route.fulfill({response,body:`${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='BASELINE'){console.info('parent-prefix');setTimeout(()=>dispatch.call(self,e),500);}else dispatch.call(self,e);};}`});});
  await page.goto('?backend=cpu');await expect(page.locator('.status')).toHaveText('Showing 1.00×',{timeout:60000});await page.getByLabel('Open story workspace').setInputFiles({name:'short-parent.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(workspace))});await page.locator('.story-map-comparison > summary').click();
  const shared=page.getByRole('region',{name:'Shared story continuations',exact:true});await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-0');await shared.locator('.story-parent-comparison > summary').click();const panel=shared.getByRole('region',{name:'Parent versus new branch probabilities',exact:true});
  await panel.getByLabel('Parent comparison chosen token').selectOption('471');
  await panel.getByRole('button',{name:'Measure parent and branch probabilities',exact:true}).click();await expect(panel.getByTestId('parent-trace-status')).toContainText('1 / 16 predictions measured.',{timeout:60000});await panel.getByRole('button',{name:'Stop parent comparison',exact:true}).click();
  const partial=await downloadStory(page,'Download parent comparison');expect(partial.complete).toBe(false);expect(partial.rows[0].parent).toHaveLength(1);expect(partial.rows[0].child).toHaveLength(0);expect(partial.differences.flat()).toHaveLength(0);
  for(const change of ['target','close']){
    const pending=page.waitForEvent('console',m=>m.text()==='parent-prefix');await panel.getByRole('button',{name:'Measure parent and branch probabilities',exact:true}).click();await pending;
    if(change==='target')await panel.getByLabel('Parent comparison chosen token').selectOption('1');else{await shared.locator('.story-parent-comparison > summary').click();await shared.locator('.story-parent-comparison > summary').click();}
    await expect(panel.locator('.parent-trace-point')).toHaveCount(0);await expect(panel.getByRole('button',{name:'Download parent comparison',exact:true})).toBeDisabled();await expect(shared.getByRole('button',{name:'Measure token-by-token interactions',exact:true})).toBeEnabled();
  }
  await panel.getByRole('button',{name:'Measure parent and branch probabilities',exact:true}).click();await expect(panel.getByTestId('parent-trace-status')).toHaveText('Parent comparison complete. 16 / 16 predictions measured.',{timeout:60000});const full=await downloadStory(page,'Download parent comparison');
  expect(full.rows.map((track:any)=>[track.parent.length,track.child.length])).toEqual([[2,6],[2,6]]);expect(full.differences.map((track:any[])=>track.length)).toEqual([2,2]);
  const tail=panel.locator('.parent-trace-point[data-side="0"][data-offset="4"]');await expect(tail).not.toHaveAttribute('data-parent');await expect(tail).not.toHaveAttribute('data-difference');await tail.click();await expect(panel.getByTestId('parent-trace-difference-0')).toHaveText('No paired value');
  const after=await downloadStory(page,'Download workspace');expect(after.nodes).toEqual(workspace.nodes);await expect(page.getByRole('alert')).toHaveCount(0);
});
