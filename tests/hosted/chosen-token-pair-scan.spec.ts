import { test } from '@playwright/test';
import { chosenScanWorkflow } from '../support/chosen-token-pair-scan';
test('hosted chosen-token pair scan ranks signed effects, previews and persists an exact-source branch',async({page},info)=>{
  await chosenScanWorkflow(page,0,info);
});
