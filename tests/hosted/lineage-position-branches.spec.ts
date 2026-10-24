import { test } from '@playwright/test';
import { trajectoryPositionBranchWorkflow } from '../support/trajectory-position-branches';
import { parentPositionBranchWorkflow } from '../support/parent-position-branches';
import { lineagePositionBranchWorkflow } from '../support/lineage-position-branches';
test('hosted cumulative rankings jump past immediate-parent tails and save another paired generation from the latest stories',async({page},info)=>{await trajectoryPositionBranchWorkflow(page);await parentPositionBranchWorkflow(page);await lineagePositionBranchWorkflow(page,info);});
