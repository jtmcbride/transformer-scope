import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { BpeTokenizer, type BpeDocument } from '../engine/bpe';
import type { ModelManifest } from '../engine/contracts';
import { tokenLabel } from '../engine/tokenizer';
import { selectionEdits } from '../continuation';
import { captureParentStories, parentCounts, parentDifferences, traceParentProbabilities, type ParentProbabilityRows, type ParentRole } from '../story-parent-probabilities';
import { checkpointKey, storyAnalysisKey, type SharedStoryGroup, type StoryWorkspace } from '../story-workspace';
import { interactionFill } from '../head-pair-heatmap';
import { ModelClient } from '../workers/client';
import { loadStoryModel, predictStory } from '../workers/story-runtime';
import { parentPositionBranchDraft, parentPositionChanges, parentPositionScore, rankParentPositions } from '../story-parent-position-ranking';
import type { PositionDifferenceOrder, TrajectoryBranchDraft } from '../story-trajectory-differences';
import '../story-parent-comparison.css';

interface Props {workspace:StoryWorkspace;group:SharedStoryGroup;manifest:ModelManifest;tokenizer:BpeDocument;disabled:boolean;onBusy:(busy:boolean)=>void;onOpenStory:(id:string)=>void;onSetupBranch:(key:string,draft:TrajectoryBranchDraft)=>void;onClearBranch:(key:string)=>void}
const empty=():ParentProbabilityRows=>[{parent:[],child:[]},{parent:[],child:[]}];
const EMPTY=empty();
const percent=(p:number)=>`${(p*100).toFixed(4)}%`;
const points=(p:number)=>{const value=Number((p*100).toFixed(4));return `${value>0?'+':''}${value.toFixed(4)} pp`;};
type Trace={key:string;rows:ParentProbabilityRows;busy:boolean;complete:boolean;status:string;error:string};
export function StoryParentComparison(props:Props){
  const [open,setOpen]=useState(false);
  return <details className="story-parent-comparison" open={open} onToggle={event=>setOpen(event.currentTarget.open)}><summary>Compare new branches with their parent stories</summary>{open&&<ParentTrace {...props}/>}</details>;
}
function ParentTrace({workspace,group,manifest,tokenizer,disabled,onBusy,onOpenStory,onSetupBranch,onClearBranch}:Props){
  const snapshot=useMemo(()=>captureParentStories(workspace,group),[workspace,group]);
  const [branch,setBranch]=useState<'original'|'edited'>('edited'),[target,setTarget]=useState(()=>group.targetTokenId??snapshot.children[0].plan.source.choice.tokenId);
  const [metric,setMetric]=useState<'probability'|'difference'>('probability'),[trace,setTrace]=useState<Trace|null>(null),[selected,setSelected]=useState(0),[focused,setFocused]=useState('');
  const [order,setOrder]=useState<PositionDifferenceOrder>('absolute');
  const key=`parent:${storyAnalysisKey([checkpointKey(manifest),snapshot,branch,target])}`,captured=useMemo(()=>structuredClone(snapshot),[key]);
  const owner=useRef<ModelClient|null>(null),epoch=useRef(0),chart=useRef<HTMLDivElement>(null);
  const current=!disabled&&trace?.key===key?trace:null,rows=current?.rows??EMPTY,busy=!!current?.busy;
  const counts=parentCounts(captured,branch),total=counts.reduce((n,row)=>n+row.parent+row.child,0),count=Math.max(...counts.flatMap(row=>[row.parent,row.child]));
  const differences=useMemo(()=>parentDifferences(rows,target),[rows,target]);
  const positionChanges=useMemo(()=>parentPositionChanges(rows,target),[rows,target]),ranked=useMemo(()=>rankParentPositions(positionChanges,order),[positionChanges,order]);
  const selectedChange=positionChanges.find(row=>row.offset===selected),pairedCount=Math.min(...counts.flatMap(row=>[row.parent,row.child]));
  const all=rows.flatMap(row=>[...row.parent,...row.child]),limit=all.reduce((n,row)=>Math.max(n,row.probability),0),differenceLimit=differences.flat().reduce((n,row)=>Math.max(n,Math.abs(row.difference)),0);
  const encoder=useMemo(()=>new BpeTokenizer(tokenizer),[tokenizer]);
  const options=useMemo(()=>tokenizer.tokens.map((_,id)=>id!==0&&id!==2&&<option key={id} value={id}>{tokenLabel(tokenizer,id)} · token {id}</option>),[tokenizer]);
  useEffect(()=>{setTrace(null);setSelected(0);setFocused('');return()=>{++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);onClearBranch(key);};},[key,onBusy,onClearBranch]);
  useEffect(()=>{if(disabled){++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);setTrace(null);onClearBranch(key);}},[disabled,onBusy,onClearBranch,key]);
  function stop(){++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);onClearBranch(key);setTrace(old=>old?.busy?{...old,busy:false,complete:false,status:'Parent comparison stopped. Results are partial.'}:old);}
  function jump(offset:number){onClearBranch(key);setSelected(offset);setMetric('difference');const point=chart.current?.querySelector<HTMLButtonElement>(`button[data-side="0"][data-offset="${offset}"]`);point?.focus({preventScroll:true});point?.scrollIntoView({block:'nearest',inline:'center'});}
  async function start(){
    if(busy||disabled||!counts.some(row=>row.child))return;
    onClearBranch(key);
    const generation=++epoch.current,valid=()=>generation===epoch.current;onBusy(true);setSelected(0);setFocused('');
    setTrace({key,rows:empty(),busy:true,complete:false,status:'Measuring parent and branch probabilities…',error:''});
    const client=new ModelClient();owner.current=client;
    try{
      await loadStoryModel(client,manifest);if(!valid())return;
      await traceParentProbabilities(captured,branch,target,manifest.config,(ids,edits)=>predictStory(client,ids,edits),(side,role,point)=>{if(valid())setTrace(old=>old?.key===key?{...old,rows:old.rows.map((track,i)=>i===side?{...track,[role]:[...track[role],point]}:track) as ParentProbabilityRows}:old);},()=>!valid());
      if(valid())setTrace(old=>old?{...old,busy:false,complete:true,status:'Parent comparison complete.'}:old);
    }catch(e){if(valid())setTrace(old=>old?{...old,busy:false,complete:false,status:'Parent comparison failed. Results are partial.',error:e instanceof Error?e.message:String(e)}:old);}
    finally{client.dispose();if(valid()){owner.current=null;onBusy(false);}}
  }
  function download(){if(!current)return;const data={format:'transformer-story-parent-probabilities',version:1,checkpoint:checkpointKey(manifest),backend:'cpu',snapshot:captured,childBranch:branch,targetTokenId:target,display:metric,rows,differences,positionChanges:ranked,positionChangeOrder:order,positionChangeDefinition:'difference = new minus parent per story; absolute score = max(abs(A), abs(B)); increase score = max(A, B); decrease score = min(A, B); only offsets measured on all four paths',complete:current.complete&&rows.every((track,i)=>track.parent.length===counts[i].parent&&track.child.length===counts[i].child),status:current.status,error:current.error||undefined,definition:'chosen-token probability on each actual saved path; difference = new branch minus parent, at matching generation offsets from the fork'};
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download='story-parent-probabilities.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),0);}
  const coordinates=rows.flatMap((track,side)=>[...new Set([...track.parent,...track.child].map(row=>row.offset))].sort((a,b)=>a-b).map(index=>`${side}:${index}`)),tabStop=coordinates.includes(focused)?focused:coordinates[0];
  function move(event:KeyboardEvent<HTMLButtonElement>,side:number,index:number){if(event.altKey||event.metaKey)return;const offsets=coordinates.filter(key=>key.startsWith(`${side}:`));let next:string|undefined;
    if(event.key==='Home')next=event.ctrlKey?coordinates[0]:offsets[0];else if(event.key==='End')next=event.ctrlKey?coordinates.at(-1):offsets.at(-1);else if(event.key==='ArrowLeft')next=`${side}:${index-1}`;else if(event.key==='ArrowRight')next=`${side}:${index+1}`;else if(event.key==='ArrowUp')next=`${side-1}:${index}`;else if(event.key==='ArrowDown')next=`${side+1}:${index}`;else return;
    event.preventDefault();if(next&&coordinates.includes(next)){const [s,i]=next.split(':');chart.current?.querySelector<HTMLButtonElement>(`button[data-side="${s}"][data-offset="${i}"]`)?.focus();}}
  function savedState(side:number,role:ParentRole,index:number){if(index<counts[side][role])return 'Not measured';if(role==='child')return captured.children[side].frame[branch]!.finish??'No saved token yet';const p=captured.parents[side],path=captured.children[side].plan.source.branch;return (p.kind==='node'?p.node.frame:p.root.frame)[path]!.finish??'No saved token yet';}
  return <section aria-label="Parent versus new branch probabilities">
    <p className="eyebrow">Follow what changed after rebranching</p><h4>How did the chosen token’s probability change?</h4>
    <p>Compare each saved parent story from its fork point with its new continuation. Every measurement uses that path’s actual preceding text, head settings, and recorded sampling draw. The chosen token stays fixed; the tokens actually generated can differ.</p>
    <div className="continuation-actions"><label>New paths<select aria-label="Parent comparison new paths" value={branch} disabled={busy||disabled} onChange={event=>setBranch(event.target.value as typeof branch)}><option value="edited">With new head changes</option><option value="original">Keep source settings</option></select></label>
      <label>Chosen next token<select aria-label="Parent comparison chosen token" value={target} disabled={disabled} onChange={event=>setTarget(Number(event.target.value))}>{options}</select></label>
      <label>Show<select aria-label="Parent comparison display" value={metric} onChange={event=>setMetric(event.target.value as typeof metric)}><option value="probability">Parent and new probability</option><option value="difference">New minus parent</option></select></label>
      <button type="button" disabled={busy||disabled||!counts.some(row=>row.child)} onClick={()=>void start()}>Measure parent and branch probabilities</button><button type="button" disabled={!busy} onClick={stop}>Stop parent comparison</button><button type="button" disabled={busy||!current} onClick={download}>Download parent comparison</button>
    </div>
    <p data-testid="parent-trace-target">Following <strong>{tokenLabel(tokenizer,target)}</strong> (token {target}) as the possible next token at every position. Words can span several model tokens; ␠ marks a space.</p>
    <p role="status" data-testid="parent-trace-status">{current?.status??'Ready to compare saved parent and branch tokens.'} {all.length} / {total} predictions measured.</p>
    {busy&&<progress aria-label="Parent probability measurement progress" value={all.length} max={total}/>}{current?.error&&<p role="alert">Could not finish: {current.error} Completed measurements are kept.</p>}
    <p className="hint">{metric==='probability'?`All four paths share 0–${percent(limit)}. Gray: parent · Purple: new branch.`:`Both stories share ±${(differenceLimit*100).toFixed(4)} pp. Orange: higher in the new branch · Blue: lower.`} pp = percentage points. Unmeasured points are hatched; unavailable saved tokens have no invented probability. Offsets align generation steps from the fork, not matching words. After the text diverges, differences include that changed context as well as head settings.</p>
    <div ref={chart} className="parent-trace-scroll" role="region" aria-label="Parent and new branch probability tracks" tabIndex={0} data-probability-limit={limit} data-difference-limit={differenceLimit} data-metric={metric}>
      {captured.children.map((node,side)=><div key={node.id} className="parent-trace-track" role="group" aria-label={`Story ${side===0?'A':'B'} parent comparison track`}><h5>{side===0?'A':'B'} · {captured.group.labels[side]}</h5><p className="hint">Parent: {node.parentId?'saved comparison':'source story'} · {node.plan.source.branch==='edited'?'changed':'source-settings'} path · starts at story step {node.plan.source.step}.</p>
        <div className="parent-trace-columns" style={{gridTemplateColumns:`repeat(${count},minmax(100px,1fr))`,minWidth:count*100}}>{Array.from({length:count},(_,offset)=>{
          const parent=rows[side].parent.find(row=>row.offset===offset),child=rows[side].child.find(row=>row.offset===offset),difference=differences[side].find(row=>row.offset===offset);
          const labels=([['parent',parent],['child',child]] as const).map(([role,point])=>`${role==='parent'?'Parent':'New'}: ${point?`${tokenLabel(tokenizer,point.source.choice.tokenId)} · ${percent(point.probability)}`:savedState(side,role,offset)}`);
          const content=<><small>Offset {offset+1} · step {node.plan.source.step+offset}</small><span>{labels[0]}</span><span className={`parent-trace-plot ${metric==='difference'?`parent-trace-signed ${difference?'':'parent-trace-no-difference'}`:''}`} aria-hidden="true">{metric==='probability'?<>{parent&&<i className="parent-trace-before" style={{height:`${limit?parent.probability/limit*96:0}%`}}/>}{!parent&&offset<counts[side].parent&&<i className="parent-trace-before parent-trace-pending" style={{height:'96%'}}/>}{!child&&offset<counts[side].child&&<i className="parent-trace-after parent-trace-pending" style={{height:'96%'}}/>}{child&&<i className="parent-trace-after" style={{height:`${limit?child.probability/limit*96:0}%`}}/>}</>:difference?<i className="parent-trace-difference" style={{height:`${differenceLimit?Math.abs(difference.difference)/differenceLimit*48:0}%`,...(difference.difference>=0?{bottom:'50%'}:{top:'50%'}),background:interactionFill(difference.difference,differenceLimit)}}/>:null}</span><span>{labels[1]}</span><strong>{difference?points(difference.difference):'No paired value'}</strong></>;
          return parent||child?<button key={offset} type="button" className="parent-trace-point" data-side={side} data-offset={offset} data-parent={parent?.probability} data-child={child?.probability} data-difference={difference?.difference} data-same-prefix={difference?.samePrefix} aria-pressed={selected===offset} disabled={busy} tabIndex={`${side}:${offset}`===tabStop?0:-1} onFocus={()=>setFocused(`${side}:${offset}`)} onKeyDown={event=>move(event,side,offset)} onClick={()=>{onClearBranch(key);setSelected(offset);}} aria-label={`Story ${side===0?'A':'B'}, offset ${offset+1}, step ${node.plan.source.step+offset}. Target ${tokenLabel(tokenizer,target)}. ${labels.join('. ')}. New minus parent ${difference?points(difference.difference):'not available'}.`}>{content}</button>:<div key={offset} className="parent-trace-unmeasured">{content}</div>;
        })}</div></div>)}
    </div>
    <p className="hint">Select a position to inspect both stories. Tab enters a measured point; arrows move between positions or stories, Enter selects. A missing parent or child value is excluded from the difference.</p>
    <section className="parent-position-ranking" aria-label="Ranked parent-to-new probability changes">
      <h5>Where did rebranching change the chosen token most?</h5>
      <label>Rank positions<select aria-label="Rank parent-to-new probability changes" value={order} onChange={event=>setOrder(event.target.value as PositionDifferenceOrder)}><option value="absolute">Largest change in either story (absolute)</option><option value="increase">Most positive change in either story</option><option value="decrease">Most negative change in either story</option></select></label>
      <p data-testid="parent-ranking-status">{positionChanges.length} / {pairedCount} offsets measured on all four paths. {current?.complete?'Complete for these saved paths.':'Partial until measurement finishes.'} Each change is new minus parent. Ranking uses the larger absolute change, most positive change, or most negative change across A and B, without rounding.</p>
      <ol aria-label="Positions ranked by parent-to-new probability change">{ranked.map(row=><li key={row.offset} data-offset={row.offset} data-score={parentPositionScore(row,order)} data-a-change={row.a.difference} data-b-change={row.b.difference}>
        <button type="button" disabled={busy||disabled} aria-current={selected===row.offset?'location':undefined} onClick={()=>jump(row.offset)} aria-label={`Jump to rebranch offset ${row.offset+1}, A step ${row.a.child.source.step}, B step ${row.b.child.source.step}, A change ${points(row.a.difference)}, B change ${points(row.b.difference)}`}>
          <strong>Offset {row.offset+1} · {order==='absolute'?`largest magnitude ${(parentPositionScore(row,order)*100).toFixed(4)} pp`:`${order==='increase'?'most positive':'most negative'} ${points(parentPositionScore(row,order))}`}</strong>
          {([row.a,row.b] as const).map((point,side)=><span key={side}>{side===0?'A':'B'} · step {point.child.source.step} · parent {percent(point.parent.probability)} → new {percent(point.child.probability)} · {points(point.difference)}</span>)}
        </button>
      </li>)}</ol>
      {!ranked.length&&<p>Measure the chosen token on all four paths to rank positions. Missing, ended, or differently targeted predictions are excluded; measured zeroes remain.</p>}
      <button type="button" disabled={busy||disabled||!selectedChange} onClick={()=>{if(selectedChange)onSetupBranch(key,parentPositionBranchDraft(captured,branch,selectedChange));}}>Set up both new branches from offset {selected+1}</button>
      <p className="hint">Jump selects this offset in both tracks. Branch again from both new continuations’ actual pre-token text, with the same pair and chosen target. You can adjust the strengths, duration, and length before generating. Parent-to-new differences include changed text as well as head settings; they do not isolate one cause.</p>
    </section>
    <div className="parent-trace-details">{captured.children.map((node,side)=>{const difference=differences[side].find(row=>row.offset===selected);return <article key={node.id} aria-label={`Story ${side===0?'A':'B'} parent comparison details`}><h5>{side===0?'A':'B'} · offset {selected+1} · step {node.plan.source.step+selected}</h5>
      <p>New minus parent: <strong data-testid={`parent-trace-difference-${side}`}>{difference?points(difference.difference):'No paired value'}</strong></p>{difference&&<p>{difference.samePrefix?'Both predictions use the same preceding text.':'The preceding text differs between these saved paths.'}</p>}
      {(['parent','child'] as const).map(role=>{const point=rows[side][role].find(row=>row.offset===selected),parent=captured.parents[side],nodeId=role==='child'?node.id:parent.kind==='node'?parent.node.id:'';return <div key={role} className="parent-trace-detail" role="group" aria-label={`${side===0?'A':'B'} ${role==='parent'?'parent':'new branch'} selected prediction`}><h6>{role==='parent'?'Parent story':'New branch'}</h6>{point?<>
        <p>Chosen-token probability: <strong data-testid={`parent-trace-${role}-${side}`}>{percent(point.probability)}</strong></p><p>Actually generated: {tokenLabel(tokenizer,point.source.choice.tokenId)} · recorded probability {percent(point.source.choice.probability)}. Recorded choice reproduced.</p>
        <details><summary>Exact preceding text and settings</summary><p className="inspection-prefix">{encoder.preview(point.source.prefix)}</p><p>{selectionEdits(point.source).map(edit=>`Layer ${edit.layer+1} · Head ${edit.head+1} · ${edit.factor}×`).join('; ')||'No head edits.'}</p><p>Temperature {point.source.settings.temperature} · top-p {point.source.settings.topP} · seed {point.source.settings.seed} · global step {point.source.step}.</p></details>
      </>:<p>{savedState(side,role,selected)}.</p>}<button type="button" disabled={busy||disabled} onClick={()=>{stop();onOpenStory(nodeId);}}>Open {side===0?'A':'B'} {role==='parent'?'parent story':'new branch'}</button></div>;})}
    </article>;})}</div>
    <p className="hint">This measures actual saved continuations, including their active edits and expiry. It does not generate or extend either story. Measurements are temporary; changing target, paths, saved inputs, or starting other analysis clears them. Download the comparison to keep its probabilities and exact captured inputs.</p>
  </section>;
}
