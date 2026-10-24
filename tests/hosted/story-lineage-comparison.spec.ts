import { test } from '@playwright/test';
import { trajectoryPositionBranchWorkflow } from '../support/trajectory-position-branches';
import { parentPositionBranchWorkflow } from '../support/parent-position-branches';
import { lineageProbabilityComparisonWorkflow } from '../support/lineage-probability-comparison';
test('hosted repeated rebranching compares original, immediate parent and latest chosen-token probabilities',async({page},info)=>{await trajectoryPositionBranchWorkflow(page);await parentPositionBranchWorkflow(page);await lineageProbabilityComparisonWorkflow(page,info);});
