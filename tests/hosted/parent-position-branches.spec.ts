import { test } from '@playwright/test';
import { trajectoryPositionBranchWorkflow } from '../support/trajectory-position-branches';
import { parentPositionBranchWorkflow } from '../support/parent-position-branches';
test('hosted parent-change ranking jumps to measured positions and saves descendants of both new continuations',async({page},info)=>{await trajectoryPositionBranchWorkflow(page);await parentPositionBranchWorkflow(page,info);});
