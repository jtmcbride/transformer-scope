import { test } from '@playwright/test';
import { trajectoryPositionBranchWorkflow } from '../support/trajectory-position-branches';
import { parentProbabilityComparisonWorkflow } from '../support/parent-probability-comparison';
test('hosted new branches compare chosen-token probabilities with their actual saved parents',async({page},info)=>{await trajectoryPositionBranchWorkflow(page);await parentProbabilityComparisonWorkflow(page,info);});
