import { expect, test } from '@playwright/test';
import { openPositionRanking, trajectoryPositionBranchWorkflow } from '../support/trajectory-position-branches';
import { downloadStory } from '../support/chosen-token-pair-scan';

test('ranks paired token effects, jumps with keyboard and touch layout, and saves nested branches from both actual paths',async({page},info)=>{test.setTimeout(180000);await trajectoryPositionBranchWorkflow(page,info);});
test('partial rankings exclude missing peers and clear branch setup on target, path, group and remeasurement changes',async({page})=>{
  test.setTimeout(180000);
  await page.route('**/model.worker-*.js',async route=>{const response=await route.fetch();await route.fulfill({response,body:`${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='BASELINE')setTimeout(()=>dispatch.call(self,e),500);else dispatch.call(self,e);};}`});});
  const {workspace,shared,trace}=await openPositionRanking(page),ranking=trace.getByRole('region',{name:'Ranked probability-effect differences',exact:true}),setup=shared.getByRole('group',{name:'Rebranch both stories from trajectory',exact:true});
  await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();await expect(trace.getByTestId('trajectory-status')).toContainText('1 / 12 tokens measured.',{timeout:60000});await trace.getByRole('button',{name:'Stop token trace',exact:true}).click();
  await expect(ranking.locator('li')).toHaveCount(0);await expect(ranking.getByRole('button',{name:/Set up both branches/})).toBeDisabled();
  async function partialSetup(){await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();await expect(ranking.locator('li')).toHaveCount(1,{timeout:60000});await expect(ranking.getByRole('button',{name:/Set up both branches/})).toBeDisabled();await trace.getByRole('button',{name:'Stop token trace',exact:true}).click();await ranking.getByRole('button',{name:'Set up both branches from offset 1',exact:true}).click();await expect(setup).toBeVisible();}
  await partialSetup();expect((await downloadStory(page,'Download token interactions')).complete).toBe(false);await trace.getByLabel('Chosen next token').selectOption(String(workspace.nodes[1].frame.edited!.choices[0].tokenId));await expect(setup).toHaveCount(0);await expect(ranking.locator('li')).toHaveCount(0);
  await partialSetup();await trace.getByLabel('Interaction trajectory paths').selectOption('original');await expect(setup).toHaveCount(0);
  await partialSetup();await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-1');await expect(setup).toHaveCount(0);
  await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();await expect(trace.getByTestId('trajectory-status')).toContainText('Token interactions complete.',{timeout:60000});
  await expect(ranking.locator('li')).toHaveCount(6);await expect(ranking.locator('li').nth(1)).toContainText('edits expired');
  await ranking.getByRole('button',{name:/Set up both branches/}).click();await expect(setup).toBeVisible();await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();await expect(setup).toHaveCount(0);await trace.getByRole('button',{name:'Stop token trace',exact:true}).click();
  const saved=await downloadStory(page,'Download workspace');expect(saved.nodes).toEqual(workspace.nodes);await expect(page.getByRole('alert')).toHaveCount(0);
});
