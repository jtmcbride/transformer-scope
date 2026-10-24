import { expect, test } from '@playwright/test';
import input from '../fixtures/story-lineage-workspace.json' with {type:'json'};
import { trajectoryPositionBranchWorkflow } from '../support/trajectory-position-branches';
import { parentPositionBranchWorkflow } from '../support/parent-position-branches';
import { lineagePositionBranchWorkflow } from '../support/lineage-position-branches';
import { downloadStory } from '../support/chosen-token-pair-scan';

test('ranks cumulative changes against the originals, jumps without inference and branches both latest stories beyond their parents',async({page},info)=>{test.setTimeout(180000);await trajectoryPositionBranchWorkflow(page);await parentPositionBranchWorkflow(page);await lineagePositionBranchWorkflow(page,info);});
test('cumulative rankings exclude partial and missing original values and clear obsolete paired branch setup',async({page})=>{
  test.setTimeout(180000);
  const workspace=structuredClone(input.workspace);
  await page.route('**/model.worker-*.js',async route=>{const response=await route.fetch();await route.fulfill({response,body:`${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='BASELINE'){console.info('cumulative-prefix');setTimeout(()=>dispatch.call(self,e),500);}else dispatch.call(self,e);};}`});});
  await page.goto('?backend=cpu');await expect(page.locator('.status')).toHaveText('Showing 1.00×',{timeout:60000});await page.getByLabel('Open story workspace').setInputFiles({name:'short-originals.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(workspace))});await page.locator('.story-map-comparison > summary').click();
  const shared=page.getByRole('region',{name:'Shared story continuations',exact:true});await shared.getByLabel('Saved shared continuation').selectOption('lineage-comparison');await shared.locator('.story-lineage-comparison > summary').click();
  const panel=shared.getByRole('region',{name:'Original, parent and latest story probabilities',exact:true}),ranking=panel.getByRole('region',{name:'Ranked original-to-latest probability changes',exact:true}),list=ranking.getByRole('list',{name:'Positions ranked by original-to-latest probability change',exact:true}),setup=shared.getByRole('group',{name:'Rebranch both stories from original comparison',exact:true});
  const measure=panel.getByRole('button',{name:'Measure original, parent and latest probabilities',exact:true}),stop=panel.getByRole('button',{name:'Stop lineage comparison',exact:true}),prepare=ranking.getByRole('button',{name:/Set up both latest branches from offset/});
  async function progress(value:number){await expect(panel.getByTestId('lineage-status')).toContainText(new RegExp(`${value} / 14 predictions measured\\.|Lineage comparison failed\\.`),{timeout:60000});if(await panel.getByRole('alert').count())throw new Error(await panel.getByRole('alert').innerText());}
  await expect(prepare).toBeDisabled();await measure.click();await progress(1);await expect(list.locator('li')).toHaveCount(0);await progress(3);await expect(list.locator('li')).toHaveCount(0);await progress(6);await stop.click();await expect(list.locator('li')).toHaveCount(1);await expect(ranking.getByTestId('lineage-ranking-status')).toContainText('1 / 1 offsets measured on both originals and both latest paths. Partial');
  const partial=await downloadStory(page,'Download lineage comparison');expect(partial.complete).toBe(false);expect(partial.positionChanges).toHaveLength(1);
  await measure.click();await expect(panel.getByTestId('lineage-status')).toHaveText('Lineage comparison complete. 14 / 14 predictions measured.',{timeout:60000});await expect(list.locator('li')).toHaveCount(1);expect((await downloadStory(page,'Download lineage comparison')).positionChanges.map((p:any)=>p.offset)).toEqual([0]);
  await prepare.click();await ranking.getByLabel('Rank original-to-latest probability changes').selectOption('decrease');await panel.getByLabel('Lineage comparison display').selectOption('recent');await expect(setup).toBeVisible();
  await panel.locator('.lineage-point[data-side="0"][data-offset="1"]').click();await expect(prepare).toBeDisabled();await expect(setup).toHaveCount(0);await expect(panel.getByTestId('lineage-cumulative-0')).toHaveText('No paired value');await expect(panel.getByTestId('lineage-recent-0')).not.toHaveText('No paired value');
  for(const change of ['target','path','close','group','remeasure','competing']){
    await measure.click();await progress(6);await stop.click();await prepare.click();await expect(setup).toBeVisible();
    if(change==='target')await panel.getByLabel('Lineage comparison chosen token').selectOption('1');
    else if(change==='path')await panel.getByLabel('Lineage comparison latest paths').selectOption('original');
    else if(change==='close'){await shared.locator('.story-lineage-comparison > summary').click();await shared.locator('.story-lineage-comparison > summary').click();}
    else if(change==='group'){await shared.getByLabel('Saved shared continuation').selectOption('parent-comparison');await shared.getByLabel('Saved shared continuation').selectOption('lineage-comparison');await shared.locator('.story-lineage-comparison > summary').click();}
    else if(change==='remeasure'){const pending=page.waitForEvent('console',m=>m.text()==='cumulative-prefix');await measure.click();await pending;await stop.click();}
    else{await shared.locator('.story-parent-comparison > summary').click();const parent=shared.getByRole('region',{name:'Parent versus new branch probabilities',exact:true}),pending=page.waitForEvent('console',m=>m.text()==='cumulative-prefix');await parent.getByRole('button',{name:'Measure parent and branch probabilities',exact:true}).click();await pending;await parent.getByRole('button',{name:'Stop parent comparison',exact:true}).click();}
    await expect(setup).toHaveCount(0);await expect(list.locator('li')).toHaveCount(0);await expect(prepare).toBeDisabled();
  }
  const saved=await downloadStory(page,'Download workspace');expect(saved.root).toEqual(workspace.root);expect(saved.nodes).toEqual(workspace.nodes);await expect(page.getByRole('alert')).toHaveCount(0);
});
