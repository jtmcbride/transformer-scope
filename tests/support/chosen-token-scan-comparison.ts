import { expect, type Page, type TestInfo } from '@playwright/test';
import input from '../fixtures/story-trajectory-workspace.json' with { type:'json' };
import reference from '../fixtures/chosen-token-scan-comparison-reference.json' with { type:'json' };
import { downloadStory } from './chosen-token-pair-scan';
export async function openChosenComparison(page:Page){
  await page.goto('?backend=cpu');await expect(page.locator('.status')).toHaveText('Showing 1.00×',{timeout:60000});
  await page.getByLabel('Open story workspace').setInputFiles({name:'paired-target.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(input.workspace))});
  await page.locator('.story-map-comparison > summary').click();
  const panel=page.getByRole('region',{name:'Compare story interaction heatmaps',exact:true});
  for(const [i,letter] of ['A','B'].entries()){
    await panel.getByLabel(`Story ${letter} path`).selectOption(JSON.stringify([reference.nodeIds[i],'edited']));
    await panel.getByLabel(`Token ${letter} position`).selectOption(String(reference.index));
  }
  await panel.getByLabel('Comparison target mode').selectOption('fixed');
  await panel.getByLabel('Shared scan target token').selectOption('471');
  await expect(panel.getByLabel('Compare head-pair effects')).toHaveValue('probability');
  return panel;
}
export async function chosenComparisonWorkflow(page:Page,info?:TestInfo){
  const panel=await openChosenComparison(page), example=reference.cases[0];
  await panel.getByRole('button',{name:'Scan both tokens',exact:true}).click();
  await expect(panel.getByTestId('comparison-map-status')).toHaveText('Heatmap comparison complete. A: 120 / 120 · B: 120 / 120 · 120 matching combinations.',{timeout:240000});
  const saved=await downloadStory(page,'Download heatmap comparison');
  expect(saved.backend).toBe('cpu');expect(saved.differenceDefinition).toBe('(p_pair_B - p_source_B) - (p_pair_A - p_source_A)');expect(saved.version).toBe(2);expect(saved.target).toEqual({kind:'fixed',tokenId:471});expect(saved.metric).toBe('probability');expect(saved.complete).toBe(true);expect(saved.sources).toEqual(reference.sources);
  for(const side of [0,1])saved.rows[side].forEach((row:any,i:number)=>{expect(row.edits).toEqual(example.rows[side][i].edits);expect(row.before).toBeCloseTo(example.rows[side][i].before,5);expect(row.after).toBeCloseTo(example.rows[side][i].after,5);expect(row.delta).toBeCloseTo(example.rows[side][i].delta,5);});
  saved.changes.forEach((row:any,i:number)=>{expect(row.difference).toBe(row.b.delta-row.a.delta);expect(row.difference).toBeCloseTo(example.changes[i].difference,5);});
  const difference=panel.getByRole('article',{name:'Probability effect difference map',exact:true});
  const cells=await difference.locator('.pair-map-cell').evaluateAll(items=>items.map(el=>({key:el.getAttribute('data-pair-key'),value:Number(el.getAttribute('data-interaction'))})));
  expect(cells).toHaveLength(240);for(const cell of cells)expect(cell.value).toBe(saved.changes.find((r:any)=>r.key===cell.key).difference);
  for(const order of ['absolute','increase','decrease']){
    await panel.getByLabel('Rank probability effect differences').selectOption(order);
    const expected=[...saved.changes].sort((a:any,b:any)=>order==='absolute'?Math.abs(b.difference)-Math.abs(a.difference):order==='increase'?b.difference-a.difference:a.difference-b.difference);
    const keys=await panel.getByRole('list',{name:'Largest probability effect differences'}).locator('li').evaluateAll(items=>items.map(el=>el.getAttribute('data-pair-key')));
    expect(keys).toEqual(expected.slice(0,8).map((r:any)=>r.key));
  }
  // Display changes and swaps reuse the same measurements.
  await panel.getByLabel('Compare head-pair effects').selectOption('interaction');
  const interactions=await downloadStory(page,'Download heatmap comparison');expect(interactions.rows).toEqual(saved.rows);expect(interactions.metric).toBe('interaction');
  interactions.changes.forEach((row:any)=>expect(row.difference).toBe(row.b.interaction-row.a.interaction));
  await panel.getByLabel('Compare head-pair effects').selectOption('probability');
  await panel.getByRole('button',{name:'Swap A and B',exact:true}).click();
  const swapped=await downloadStory(page,'Download heatmap comparison');expect(swapped.rows).toEqual([saved.rows[1],saved.rows[0]]);expect(swapped.target).toEqual(saved.target);
  swapped.changes.forEach((row:any,i:number)=>expect(row.difference).toBeCloseTo(-saved.changes[i].difference,14));
  await panel.getByRole('button',{name:'Swap A and B',exact:true}).click();
  await panel.getByLabel('Rank probability effect differences').selectOption('absolute');
  const best=[...saved.changes].sort((a:any,b:any)=>Math.abs(b.difference)-Math.abs(a.difference))[0];
  const selected=difference.locator(`button[data-pair-key="${best.key}"]`).first();await selected.focus();await page.keyboard.press('Enter');
  expect(parseFloat(await panel.getByTestId('compared-interaction-difference').innerText())/100).toBeCloseTo(best.difference,5);
  const shared=panel.getByRole('region',{name:'Shared story continuations',exact:true});
  await expect(shared.getByRole('group',{name:'Chosen-token shared pair preview'})).toHaveAttribute('data-target-token','471');
  for(const side of [0,1])await expect(shared.getByTestId(`shared-target-preview-${side}`)).toBeVisible();
  await page.setViewportSize({width:390,height:844});expect(await panel.evaluate(el=>el.scrollWidth<=el.clientWidth+1&&el.getBoundingClientRect().right<=innerWidth)).toBe(true);
  if(info?.project.use.hasTouch){expect(await page.evaluate(()=>matchMedia('(pointer:coarse)').matches)).toBe(true);for(const control of [panel.getByLabel('Shared scan target token'),panel.getByRole('button',{name:'Scan both tokens',exact:true})])expect((await control.boundingBox())!.height).toBeGreaterThanOrEqual(44);}
  if(info){await difference.screenshot({path:info.outputPath('chosen-token-effect-difference-mobile.png')});await shared.getByRole('group',{name:'Chosen-token shared pair preview'}).screenshot({path:info.outputPath('chosen-token-shared-preview-mobile.png')});}
  await page.setViewportSize({width:1440,height:1100});
  await panel.getByRole('button',{name:'Set up both continuations',exact:true}).click();
  await expect(shared.getByLabel('Shared head-change duration')).toBeFocused();
  await shared.getByLabel('Shared continuation token limit').fill('4');
  await shared.getByRole('button',{name:'Branch both stories with this pair',exact:true}).click();
  await expect(panel.getByLabel('Shared scan target token')).toBeDisabled();
  await expect(shared.getByTestId('shared-continuation-status')).toHaveText('Shared continuations complete.',{timeout:60000});
  const record=await downloadStory(page,'Download shared comparison'),expected=example.branches.absolute.continuation;
  record.nodes.forEach((node:any,side:number)=>{expect(node.parentId).toBe(reference.nodeIds[side]);expect(node.plan.source).toEqual(reference.sources[side]);expect([node.plan.edit,...node.plan.additionalEdits]).toEqual([expected.plans[side].edit,...expected.plans[side].additionalEdits]);
    for(const branch of ['original','edited'] as const){expect(node.frame[branch].tokenIds).toEqual(expected.frames[side][branch].tokenIds);node.frame[branch].choices.forEach((choice:any,i:number)=>expect(choice.probability).toBeCloseTo(expected.frames[side][branch].choices[i].probability,5));}
  });
  const workspace=await downloadStory(page,'Download workspace');expect(workspace.root).toEqual(input.workspace.root);expect(workspace.nodes.slice(0,input.workspace.nodes.length)).toEqual(input.workspace.nodes);expect(workspace.nodes).toHaveLength(input.workspace.nodes.length+2);
  await page.reload();await page.locator('.story-map-comparison > summary').click();
  await expect(shared.getByTestId('shared-continuation-status')).toHaveText('Shared continuations complete.');expect((await downloadStory(page,'Download shared comparison')).nodes).toEqual(record.nodes);
  await expect(page.getByRole('alert')).toHaveCount(0);
}
