import { test } from '@playwright/test';
import { chosenComparisonWorkflow } from '../support/chosen-token-scan-comparison';
test('hosted same-target pair comparison ranks differences and saves both continuations',async({page},info)=>{await chosenComparisonWorkflow(page,info);});
