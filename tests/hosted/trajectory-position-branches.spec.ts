import { test } from '@playwright/test';
import { trajectoryPositionBranchWorkflow } from '../support/trajectory-position-branches';
test('hosted ranked token differences jump to both computations and create persistent shared rebranches',async({page},info)=>{await trajectoryPositionBranchWorkflow(page,info);});
