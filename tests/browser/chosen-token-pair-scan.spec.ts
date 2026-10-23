import { expect, test } from '@playwright/test';
import { chosenScanWorkflow, openChosenScan } from '../support/chosen-token-pair-scan';
for (const side of [0,1]) test(`chosen-token pair scan ranks encouragement and suppression and saves a real story from ${side ? 'B' : 'A'}`,async({page},info)=>{
  test.setTimeout(240000); await chosenScanWorkflow(page,side,info);
});
test('changing a chosen target cancels incomplete scans and clears a selected preview',async({page})=>{
  test.setTimeout(150000);
  await page.route('**/model.worker-*.js',async route=>{const response=await route.fetch();await route.fulfill({response,body:`${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='COMPARE'){console.info('chosen-pair-compare');setTimeout(()=>dispatch.call(self,e),150);}else dispatch.call(self,e);};}`});});
  const {scan}=await openChosenScan(page),list=scan.getByRole('list',{name:'Ranked head pair changes'});
  await scan.getByRole('button',{name:'Scan head pairs',exact:true}).click();
  await expect(list.locator(':scope > li').first()).toBeVisible({timeout:60000});
  await scan.getByRole('button',{name:'Stop pair scan',exact:true}).click();
  await list.locator(':scope > li').first().getByRole('button').click();
  await expect(page.getByRole('region',{name:'Scanned head probability preview',exact:true})).toBeVisible();
  await scan.getByLabel('Pair scan target token').selectOption('1');
  await expect(list).toHaveCount(0); await expect(page.getByRole('region',{name:'Scanned head probability preview',exact:true})).toHaveCount(0);
  const pending=page.waitForEvent('console',m=>m.text()==='chosen-pair-compare');
  await scan.getByRole('button',{name:'Scan head pairs',exact:true}).click(); await pending;
  await scan.getByLabel('Pair scan target token').selectOption('471');
  await expect(scan.getByTestId('pair-scan-status')).toHaveText('Ready to scan. 0 / 120 combinations measured.');
  await expect(page.getByRole('button',{name:'Scan all heads',exact:true})).toBeEnabled();
  await page.getByRole('button',{name:'Next token →',exact:true}).click();
  await expect(page.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced',{timeout:60000});
  await expect(scan.getByLabel('Pair scan target token')).toHaveValue('471');
  await expect(list).toHaveCount(0); await expect(page.getByRole('alert')).toHaveCount(0);
});
