import { openStoryAnalysis } from './story-analysis';
import { readFile } from 'node:fs/promises';
import { expect, type Page, type TestInfo } from '@playwright/test';
import input from '../fixtures/story-trajectory-workspace.json' with { type: 'json' };
import reference from '../fixtures/chosen-token-pair-scan-reference.json' with { type: 'json' };
export async function downloadStory(page: Page, name: string) {
  const wait = page.waitForEvent('download'); await page.getByRole('button',{name,exact:true}).click();
  return JSON.parse(await readFile((await (await wait).path())!,'utf8'));
}
export async function openChosenScan(page: Page, side = 0) {
  const example = reference.cases[side];
  await page.goto('?backend=cpu'); await expect(page.locator('.status')).toHaveText('Showing 1.00×',{timeout:60000});
  await page.getByLabel('Open story workspace').setInputFiles({name:'trajectory.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(input.workspace))});
  await page.locator('.story-map-comparison > summary').click();
  const shared = page.getByRole('region',{name:'Shared story continuations',exact:true});
  await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-0');
  const trace = shared.getByRole('region',{name:'Head-pair interaction along the stories',exact:true});
  await trace.getByLabel('Interaction trajectory target mode').selectOption('fixed');
  // The trace offers saved vocabulary tokens; the scanner additionally permits ungenerated EOS.
  await trace.getByLabel('Chosen next token').selectOption(String(side ? 471 : example.targetTokenId));
  await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();
  await expect(trace.getByTestId('trajectory-status')).toContainText('Token interactions complete.',{timeout:60000});
  await trace.locator('.trajectory-point[data-track="0"][data-offset="1"]').click();
  await trace.getByRole('button',{name:`Scan head pairs for ${side ? 'B' : 'A'}’s chosen token`,exact:true}).click();
  await expect(page.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced',{timeout:60000});
  expect(parseFloat(await page.getByTestId('selected-token-probability').innerText())/100).toBeCloseTo(example.source.choice.probability,5);
  await openStoryAnalysis(page);
  const scan = page.getByRole('region',{name:'Scan head pairs at selected token',exact:true});
  if (side) await scan.getByLabel('Pair scan target token').selectOption(String(example.targetTokenId));
  await expect(scan.getByLabel('Pair scan target token')).toHaveValue(String(example.targetTokenId));
  return { scan, example, trace };
}
export async function chosenScanWorkflow(page: Page, side: number, info?: TestInfo) {
  const { scan,example } = await openChosenScan(page,side), order = side ? 'decrease' : 'increase';
  await expect(scan.getByLabel('Sort pair scan results')).toHaveValue('increase');
  await scan.getByRole('button',{name:'Scan head pairs',exact:true}).click();
  await expect(page.getByRole('button',{name:'Compare from this token',exact:true})).toBeDisabled();
  await expect(scan.getByTestId('pair-scan-status')).toHaveText('Pair scan complete. 120 / 120 combinations measured.',{timeout:150000});
  await scan.getByRole('button',{name:'Show all 120 pairs',exact:true}).click();
  const list = scan.getByRole('list',{name:'Ranked head pair changes'});
  for (const mode of ['increase','decrease'] as const) {
    await scan.getByLabel('Sort pair scan results').selectOption(mode);
    const sorted = [...example.rows].sort((a,b) => mode === 'increase' ? b.delta-a.delta : a.delta-b.delta);
    const values = await list.locator(':scope > li').evaluateAll(items => items.map(el => ({a:el.getAttribute('data-pair-first'),b:el.getAttribute('data-pair-second'),after:el.querySelector('[data-testid="pair-scan-after"]')!.textContent,delta:el.querySelector('[data-testid="pair-scan-delta"]')!.textContent})));
    expect(values).toHaveLength(120);
    values.forEach((row,i) => {
      const expected=sorted[i], key=(edit:typeof expected.edits[number])=>`${edit.layer}:${edit.head}:${edit.factor}`;
      expect(row.a).toBe(key(expected.edits[0])); expect(row.b).toBe(key(expected.edits[1]));
      expect(parseFloat(row.after!)/100).toBeCloseTo(expected.after,5); expect(parseFloat(row.delta!)/100).toBeCloseTo(expected.delta,5);
    });
  }
  await scan.getByLabel('Sort pair scan results').selectOption(order);
  await list.locator(':scope > li').first().getByRole('button').click();
  const expected=example[order], row=example.rows.find(r=>JSON.stringify(r.edits)===JSON.stringify([expected.plan.edit,...expected.plan.additionalEdits]))!;
  const preview=page.getByRole('region',{name:'Scanned head probability preview',exact:true});
  await expect(preview.getByTestId('combined-probability')).toBeVisible({timeout:60000});
  const selected=preview.getByRole('list',{name:'Selected token probability'}).locator('li');
  await expect(selected).toHaveAttribute('data-token-id',String(example.targetTokenId));
  expect(parseFloat(await selected.getByTestId('probability-before').innerText())/100).toBeCloseTo(row.before,5);
  expect(parseFloat(await preview.getByTestId('combined-probability').innerText())/100).toBeCloseTo(row.after,5);
  await expect(preview.getByTestId('branch-choice-explanation')).toContainText('chooses');
  await scan.getByRole('button',{name:'Show strongest eight pairs',exact:true}).click();
  await page.setViewportSize({width:390,height:844});
  expect(await scan.evaluate(el=>el.scrollWidth<=el.clientWidth+1&&el.getBoundingClientRect().right<=innerWidth)).toBe(true);
  if (info) await scan.screenshot({path:info.outputPath('chosen-token-ranked-pairs.png')});
  await page.setViewportSize({width:1440,height:1100});
  await page.getByLabel('Branch new-token limit').fill('4');
  await page.getByRole('button',{name:'Compare from this token',exact:true}).click();
  await expect(page.locator('.fork-status')).toHaveText('Branch comparison complete.',{timeout:60000});
  const saved=await downloadStory(page,'Download branch comparison');
  expect(saved.plan.source).toEqual(example.source); expect(saved.plan.edit).toEqual(expected.plan.edit); expect(saved.plan.additionalEdits).toEqual(expected.plan.additionalEdits);
  for (const branch of ['original','edited'] as const) {
    expect(saved.frame[branch].tokenIds).toEqual(expected[branch].tokenIds);
    saved.frame[branch].choices.forEach((choice:{probability:number},i:number)=>expect(choice.probability).toBeCloseTo(expected[branch].choices[i].probability,5));
  }
  const workspace=await downloadStory(page,'Download workspace');
  expect(workspace.root).toEqual(input.workspace.root); expect(workspace.nodes.slice(0,input.workspace.nodes.length)).toEqual(input.workspace.nodes);
  expect(workspace.nodes.at(-1).parentId).toBe(example.nodeId); expect(workspace.nodes.at(-1).plan.source).toEqual(example.source);
  await page.reload(); await expect(page.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced',{timeout:60000});
  await expect(page.locator('.fork-status')).toHaveText('Branch comparison complete.');
  await expect(page.getByRole('alert')).toHaveCount(0);
}
