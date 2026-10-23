import { readFile } from 'node:fs/promises';
import { expect, test, type Page } from '@playwright/test';
import input from '../fixtures/story-trajectory-workspace.json' with {type:'json'};
import reference from '../fixtures/story-interaction-trajectories-reference.json' with {type:'json'};
import fixed from '../fixtures/story-fixed-token-trajectories-reference.json' with {type:'json'};

async function download(page:Page,name:string){
  const waiting=page.waitForEvent('download');await page.getByRole('button',{name,exact:true}).click();
  return JSON.parse(await readFile((await (await waiting).path())!,'utf8'));
}
async function open(page:Page,workspace:unknown=input.workspace){
  await page.goto('?backend=cpu');await expect(page.locator('.status')).toHaveText('Showing 1.00×',{timeout:60000});
  await page.getByLabel('Open story workspace').setInputFiles({name:'trajectory-workspace.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(workspace))});
  await page.locator('.story-map-comparison > summary').click();
  const shared=page.getByRole('region',{name:'Shared story continuations',exact:true});
  await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-0');
  return {shared,trace:shared.getByRole('region',{name:'Head-pair interaction along the stories',exact:true})};
}
test('traces exact generated targets on both paths against PyTorch, including reversals, expiry, selection and export',async({page},info)=>{
  test.setTimeout(180000);const {shared,trace}=await open(page),before=await download(page,'Download workspace');
  async function measure(caseIndex:number,branch:'original'|'edited'){
    await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();
    await expect(trace.getByTestId('trajectory-status')).toHaveText('Token interactions complete. 12 / 12 tokens measured.',{timeout:60000});
    const record=await download(page,'Download token interactions');expect(record.complete).toBe(true);expect(record.branch).toBe(branch);expect(record.backend).toBe('cpu');expect(record.checkpoint).toContain(reference.modelRevision);
    expect(record.nodes).toEqual(before.nodes.filter((node:any)=>record.group.nodeIds.includes(node.id)));
    for(const side of [0,1]){
      const expected=reference.cases[caseIndex][branch][side];expect(record.rows[side]).toHaveLength(expected.length);
      record.rows[side].forEach((point:any,i:number)=>{const r=expected[i];expect([point.index,point.step,point.tokenId,point.active]).toEqual([r.index,r.step,r.tokenId,r.active]);
        for(const key of ['before','after'] as const)expect(point[key]).toBeCloseTo(r[key],5);
        point.individualProbabilities.forEach((p:number,j:number)=>expect(p).toBeCloseTo(r.individualProbabilities[j],5));
        expect(Math.abs(point.interaction-r.interaction)).toBeLessThanOrEqual(4*5e-6);
      });
    }
    const values=await trace.locator('.trajectory-point').evaluateAll(nodes=>nodes.map(node=>Number(node.getAttribute('data-interaction'))));
    expect(values).toEqual(record.rows.flat().map((row:any)=>row.interaction));
    expect(Number(await trace.getByRole('region',{name:'Scrollable token interaction tracks',exact:true}).getAttribute('data-color-limit'))).toBe(Math.max(...values.map(Math.abs)));
    return record;
  }
  const edited=await measure(0,'edited');
  await expect(trace.locator('.trajectory-point[data-track="0"][data-offset="1"]')).toContainText('Growing');
  const reversed=trace.locator('.trajectory-point[data-track="0"][data-offset="2"]');await expect(reversed).toContainText('Reversed');
  await reversed.focus();await page.keyboard.press('Enter');await expect(trace.getByRole('button',{pressed:true})).toHaveCount(2);
  await expect(trace.getByRole('article',{name:'Story A selected interaction',exact:true})).toContainText('story step 9');
  await page.keyboard.press('ArrowDown');await expect(trace.locator('.trajectory-point[data-track="1"][data-offset="2"]')).toBeFocused();
  await page.keyboard.press('End');await page.keyboard.press('Enter');await expect(trace.getByRole('article',{name:'Story B selected interaction',exact:true})).toContainText('story step 12');
  await page.keyboard.press('Home');await expect(trace.locator('.trajectory-point[data-track="1"][data-offset="0"]')).toBeFocused();
  await trace.screenshot({path:info.outputPath('interaction-tracks.png')});
  await page.setViewportSize({width:390,height:844});expect(await trace.evaluate(el=>el.scrollWidth<=el.clientWidth+1&&el.getBoundingClientRect().right<=innerWidth)).toBe(true);
  const scroll=trace.getByRole('region',{name:'Scrollable token interaction tracks',exact:true});expect(await scroll.evaluate(el=>el.scrollWidth>el.clientWidth)).toBe(true);
  const size=await reversed.boundingBox();expect(size!.width).toBeGreaterThanOrEqual(44);expect(size!.height).toBeGreaterThanOrEqual(44);
  await trace.screenshot({path:info.outputPath('interaction-tracks-mobile.png')});await page.setViewportSize({width:1440,height:1100});
  await trace.getByLabel('Interaction trajectory paths').selectOption('original');await expect(trace.locator('.trajectory-point')).toHaveCount(0);await measure(0,'original');
  await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-1');await expect(trace.locator('.trajectory-point')).toHaveCount(0);const expired=await measure(1,'edited');
  expect(expired.rows.flat().filter((row:any)=>row.index>0).every((row:any)=>!row.active&&row.interaction===0)).toBe(true);await expect(trace.locator('.trajectory-point[data-active="false"]')).toHaveCount(10);
  await expect(trace.locator('.trajectory-point[data-active="false"]').first()).toContainText('Expired');
  const after=await download(page,'Download workspace');expect(after.root).toEqual(before.root);expect(after.nodes).toEqual(before.nodes);expect(after.sharedForks).toEqual(before.sharedForks);
  await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-0');await measure(0,'edited');
  await trace.getByRole('button',{name:'Open A comparison to inspect step 7',exact:true}).click();await page.getByRole('button',{name:/^Inspect with head change token 8:/}).click();
  await expect(page.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced',{timeout:60000});expect(parseFloat(await page.getByTestId('selected-token-probability').innerText())/100).toBeCloseTo(edited.nodes[0].frame.edited.choices[1].probability,5);
  await page.reload();await page.locator('.story-map-comparison > summary').click();await expect(trace.locator('.trajectory-point')).toHaveCount(0);await expect(shared.getByTestId('shared-continuation-status')).toHaveText('Shared continuations complete.');
});

test('partial traces exclude competing work, keep unmeasured tokens missing, and cancel on resume, close and reset',async({page})=>{
  test.setTimeout(150000);const workspace=structuredClone(input.workspace);
  for(const node of workspace.nodes.filter(node=>workspace.sharedForks[0].nodeIds.includes(node.id))){node.frame.step=2;for(const branch of ['original','edited'] as const){node.frame[branch].choices.splice(2);node.frame[branch].tokenIds.splice(node.plan.source.prefix.length+2);delete (node.frame[branch] as {finish?:string}).finish;}}
  await page.route('**/model.worker-*.js',async route=>{const response=await route.fetch();await route.fulfill({response,body:`${await response.text()}\n{let n=0;const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='BASELINE'){console.info('trajectory-prefix-'+(++n));setTimeout(()=>dispatch.call(self,e),600);}else dispatch.call(self,e);};}`});});
  const {shared,trace}=await open(page,workspace);
  await shared.getByRole('button',{name:'Open A comparison',exact:true}).click();await page.getByRole('button',{name:/^Inspect with head change token 8:/}).click();await expect(page.getByTestId('selected-token-result')).toContainText('Recorded choice reproduced',{timeout:60000});
  await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();
  for(const name of ['Step both comparisons','Play both comparisons'])await expect(shared.getByRole('button',{name,exact:true})).toBeDisabled();
  await expect(page.getByRole('button',{name:'Scan both tokens',exact:true})).toBeDisabled();await expect(page.getByRole('button',{name:'Scan head pairs',exact:true})).toBeDisabled();
  await expect(shared.getByLabel('Saved shared continuation')).toBeDisabled();await expect(trace.getByLabel('Interaction trajectory paths')).toBeDisabled();
  await expect(trace.getByTestId('trajectory-status')).toContainText('1 / 4 tokens measured.',{timeout:60000});
  await trace.getByRole('button',{name:'Stop token trace',exact:true}).click();const partial=await download(page,'Download token interactions');expect(partial.complete).toBe(false);expect(partial.rows.map((r:any[])=>r.length)).toEqual([1,0]);
  await expect(trace.locator('.trajectory-pending')).toHaveCount(3);await expect(trace.getByTestId('trajectory-status')).toContainText('Results are partial.');
  await trace.getByLabel('Interaction trajectory target mode').selectOption('fixed');
  await trace.getByLabel('Chosen next token').selectOption(String(fixed.targets[0].tokenId));
  for(const change of ['token','mode']){
    const pending=page.waitForEvent('console',m=>m.text()==='trajectory-prefix-1');await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();await pending;
    if(change==='token')await trace.getByLabel('Chosen next token').selectOption(String(workspace.nodes[1].frame.edited.choices[0].tokenId));
    else await trace.getByLabel('Interaction trajectory target mode').selectOption('generated');
    await expect(trace.locator('.trajectory-point')).toHaveCount(0);await expect(trace.getByRole('button',{name:'Download token interactions',exact:true})).toBeDisabled();await expect(shared.getByRole('button',{name:'Play both comparisons',exact:true})).toBeEnabled();
  }
  await shared.getByRole('button',{name:'Play both comparisons',exact:true}).click();await expect(trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true})).toBeDisabled();await expect(trace.locator('.trajectory-point')).toHaveCount(0);
  await expect(trace.getByRole('button',{name:'Download token interactions',exact:true})).toBeDisabled();
  await expect(shared.getByTestId('shared-continuation-status')).toHaveText('Shared continuations complete.',{timeout:60000});
  const saved=await download(page,'Download workspace');expect(saved.root).toEqual(workspace.root);expect(saved.nodes[0]).toEqual(workspace.nodes[0]);expect(saved.nodes.slice(3)).toEqual(workspace.nodes.slice(3));
  for(const action of ['close','reset']){
    const pending=page.waitForEvent('console',m=>m.text()==='trajectory-prefix-1');await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();await pending;
    if(action==='close'){await page.locator('.story-map-comparison > summary').click();await page.locator('.story-map-comparison > summary').click();}
    else{await page.getByRole('button',{name:'Reset story',exact:true}).click();await expect(page.locator('.story-map-comparison')).toHaveCount(0);await open(page,saved);}
    await expect(trace.getByTestId('trajectory-status')).toHaveText('Ready to measure token interactions. 0 / 12 tokens measured.');await expect(page.getByRole('button',{name:'Scan both tokens',exact:true})).toBeEnabled();
  }
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('pins a token across both stories with shared probability scales and measurable prediction alternatives',async({page},info)=>{
  test.setTimeout(240000);let calls=0;page.on('console',m=>{if(m.text()==='fixed-prefix')calls++;});
  await page.route('**/model.worker-*.js',async route=>{const response=await route.fetch();await route.fulfill({response,body:`${await response.text()}\n{const dispatch=self.onmessage;self.onmessage=e=>{if(e.data?.command==='BASELINE')console.info('fixed-prefix');dispatch.call(self,e);};}`});});
  const {shared,trace}=await open(page),before=await download(page,'Download workspace'),target=fixed.targets[0];
  await trace.getByLabel('Interaction trajectory target mode').selectOption('fixed');
  const generated=new Set(before.nodes.flatMap((node:any)=>[...node.frame.original.choices,...node.frame.edited.choices].map((choice:any)=>choice.tokenId)));
  const promptOnly=before.root.tokens.find((id:number)=>id!==0&&id!==2&&!generated.has(id));expect(promptOnly).toBeDefined();
  await trace.getByLabel('Chosen next token').selectOption(String(promptOnly));await expect(trace.getByLabel('Chosen next token')).toHaveValue(String(promptOnly));
  await trace.getByLabel('Chosen next token').selectOption(String(target.tokenId));
  async function measure(caseIndex:number,branch:'original'|'edited'){
    await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();await expect(trace.getByTestId('trajectory-status')).toHaveText('Token interactions complete. 12 / 12 tokens measured.',{timeout:60000});
    const record=await download(page,'Download token interactions');expect(record.version).toBe(2);expect(record.target).toEqual({kind:'fixed',tokenId:target.tokenId});expect(record.complete).toBe(true);
    for(const side of [0,1])record.rows[side].forEach((point:any,i:number)=>{const r=target.cases[caseIndex][branch][side][i];expect(point.targetTokenId).toBe(target.tokenId);expect(point.tokenId).toBe(r.tokenId);
      expect(point.recordedProbability).toBe(record.nodes[side].frame[branch].choices[i].probability);
      for(const field of ['before','after'])expect(point[field]).toBeCloseTo((r as any)[field],5);point.individualProbabilities.forEach((p:number,j:number)=>expect(p).toBeCloseTo(r.individualProbabilities[j],5));expect(Math.abs(point.interaction-r.interaction)).toBeLessThanOrEqual(4*5e-6);
    });
    expect(record.rows.flat().some((point:any)=>point.tokenId!==point.targetTokenId)).toBe(true);return record;
  }
  const edited=await measure(0,'edited'),count=calls;
  await trace.getByLabel('Interaction trajectory display').selectOption('probability');const chart=trace.getByRole('region',{name:'Scrollable token interaction tracks',exact:true});await expect(chart).toHaveAttribute('data-metric','probability');
  const limit=Math.max(...edited.rows.flat().flatMap((p:any)=>[p.before,p.after]));expect(Number(await chart.getAttribute('data-probability-limit'))).toBe(limit);await expect(trace.locator('.trajectory-probability')).toHaveCount(12);
  await trace.locator('.trajectory-point[data-track="0"][data-offset="1"]').focus();await page.keyboard.press('Enter');await expect(trace.getByRole('button',{pressed:true})).toHaveCount(2);
  await expect(trace.getByTestId('trajectory-generated-0')).toContainText('Actually generated:');expect(parseFloat(await trace.getByTestId('trajectory-after-0').innerText())/100).toBeCloseTo(edited.rows[0][1].after,6);
  await trace.screenshot({path:info.outputPath('chosen-token-probabilities.png')});await page.setViewportSize({width:390,height:844});expect(await trace.evaluate(el=>el.scrollWidth<=el.clientWidth+1&&el.getBoundingClientRect().right<=innerWidth)).toBe(true);expect(await chart.evaluate(el=>el.scrollWidth>el.clientWidth)).toBe(true);
  for(const name of ['Chosen next token','Interaction trajectory target mode'])expect((await trace.getByLabel(name).boundingBox())!.height).toBeGreaterThanOrEqual(44- (info.project.use.hasTouch?0:10));
  await trace.screenshot({path:info.outputPath('chosen-token-mobile.png')});await page.setViewportSize({width:1440,height:1100});
  await trace.getByLabel('Interaction trajectory display').selectOption('interaction');expect((await download(page,'Download token interactions')).rows).toEqual(edited.rows);expect(calls).toBe(count);
  await trace.getByLabel('Interaction trajectory paths').selectOption('original');await expect(trace.locator('.trajectory-point')).toHaveCount(0);const original=await measure(0,'original');
  const alternative=original.rows[0][0].alternatives.find((candidate:any)=>candidate.tokenId!==target.tokenId)!;
  await trace.getByRole('button',{name:new RegExp('token '+alternative.tokenId+' from A$')}).click();await expect(trace.getByLabel('Chosen next token')).toHaveValue(String(alternative.tokenId));await expect(trace.locator('.trajectory-point')).toHaveCount(0);
  await trace.getByRole('button',{name:'Measure token-by-token interactions',exact:true}).click();await expect(trace.getByTestId('trajectory-status')).toContainText('Token interactions complete.',{timeout:60000});
  const pinned=await download(page,'Download token interactions');expect(pinned.rows[0][0].before).toBeCloseTo(alternative.probability,12);expect(pinned.rows.flat().every((point:any)=>point.targetTokenId===alternative.tokenId)).toBe(true);
  await shared.getByLabel('Saved shared continuation').selectOption('trajectory-group-1');await trace.getByLabel('Interaction trajectory target mode').selectOption('fixed');await trace.getByLabel('Chosen next token').selectOption(String(target.tokenId));const expired=await measure(1,'edited');expect(expired.rows.flat().filter((p:any)=>p.index>0).every((p:any)=>p.interaction===0&&!p.active)).toBe(true);
  const after=await download(page,'Download workspace');expect(after.root).toEqual(before.root);expect(after.nodes).toEqual(before.nodes);expect(after.sharedForks).toEqual(before.sharedForks);
  await page.reload();await page.locator('.story-map-comparison > summary').click();await expect(trace.getByLabel('Interaction trajectory target mode')).toHaveValue('generated');await expect(trace.locator('.trajectory-point')).toHaveCount(0);
});
