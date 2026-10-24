import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { BpeTokenizer, type BpeDocument } from '../engine/bpe';
import type { ModelManifest } from '../engine/contracts';
import { tokenLabel } from '../engine/tokenizer';
import { selectionEdits } from '../continuation';
import { captureLineageStories, lineageCounts, originalDifferences, traceLineageProbabilities, type LineageProbabilityRows, type LineageRole } from '../story-lineage-probabilities';
import { parentDifferences } from '../story-parent-probabilities';
import { checkpointKey, type SharedStoryGroup, type StoryWorkspace } from '../story-workspace';
import { interactionFill } from '../head-pair-heatmap';
import { ModelClient } from '../workers/client';
import { loadStoryModel, predictStory } from '../workers/story-runtime';
import '../story-lineage-comparison.css';

interface Props { workspace:StoryWorkspace;group:SharedStoryGroup;manifest:ModelManifest;tokenizer:BpeDocument;disabled:boolean;onBusy:(value:boolean)=>void;onOpenStory:(id:string)=>void }
const empty=():LineageProbabilityRows=>[{original:[],parent:[],child:[]},{original:[],parent:[],child:[]}],EMPTY=empty();
const roles=['original','parent','child'] as const;
const label=(role:LineageRole)=>role==='original'?'Original':role==='parent'?'Parent':'Latest';
const percent=(p:number)=>`${(p*100).toFixed(4)}%`;
const points=(p:number)=>{const value=Number((p*100).toFixed(4));return `${value>0?'+':''}${value.toFixed(4)} pp`;};
type Metric='probability'|'recent'|'cumulative';
type Trace={key:string;rows:LineageProbabilityRows;busy:boolean;complete:boolean;status:string;error:string};
export function StoryLineageComparison(props:Props){
  const [open,setOpen]=useState(false);
  return <details className="story-lineage-comparison" open={open} onToggle={event=>setOpen(event.currentTarget.open)}><summary>Compare latest, parent, and original stories</summary>{open&&<LineageTrace {...props}/>}</details>;
}
function LineageTrace({workspace,group,manifest,tokenizer,disabled,onBusy,onOpenStory}:Props){
  const snapshot=useMemo(()=>captureLineageStories(workspace,group),[workspace,group]);
  const [branch,setBranch]=useState<'original'|'edited'>('edited'),[target,setTarget]=useState(()=>group.targetTokenId??snapshot.children[0].plan.source.choice.tokenId);
  const [metric,setMetric]=useState<Metric>('probability'),[trace,setTrace]=useState<Trace|null>(null),[selected,setSelected]=useState(0),[focused,setFocused]=useState('');
  const modelKey=checkpointKey(manifest),key=useMemo(()=>JSON.stringify([modelKey,snapshot,branch,target]),[modelKey,snapshot,branch,target]);
  const owner=useRef<ModelClient|null>(null),epoch=useRef(0),chart=useRef<HTMLDivElement>(null);
  const current=!disabled&&trace?.key===key?trace:null,rows=current?.rows??EMPTY,busy=!!current?.busy;
  const counts=lineageCounts(snapshot,branch),total=counts.reduce((sum,track)=>sum+track.original+track.parent+track.child,0),count=Math.max(...counts.map(track=>track.child));
  const recent=useMemo(()=>parentDifferences(rows,target),[rows,target]),cumulative=useMemo(()=>originalDifferences(rows,target,'child'),[rows,target]),earlier=useMemo(()=>originalDifferences(rows,target,'parent'),[rows,target]);
  const all=rows.flatMap(track=>roles.flatMap(role=>track[role])),probabilityLimit=all.reduce((max,p)=>Math.max(max,p.probability),0);
  const differenceLimit=[...recent.flat(),...cumulative.flat()].reduce((max,p)=>Math.max(max,Math.abs(p.difference)),0);
  const encoder=useMemo(()=>new BpeTokenizer(tokenizer),[tokenizer]),options=useMemo(()=>tokenizer.tokens.map((_,id)=>id!==0&&id!==2&&<option key={id} value={id}>{tokenLabel(tokenizer,id)} · token {id}</option>),[tokenizer]);
  useEffect(()=>{setTrace(null);setSelected(0);setFocused('');return()=>{++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);};},[key,onBusy]);
  useEffect(()=>{if(disabled){++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);setTrace(null);}},[disabled,onBusy]);
  function stop(){++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);setTrace(old=>old?.busy?{...old,busy:false,complete:false,status:'Lineage comparison stopped. Results are partial.'}:old);}
  async function start(){
    if(busy||disabled||!count)return;
    const generation=++epoch.current,valid=()=>generation===epoch.current;setTrace({key,rows:empty(),busy:true,complete:false,status:'Measuring original, parent and latest probabilities…',error:''});setSelected(0);setFocused('');onBusy(true);
    const client=new ModelClient();owner.current=client;
    try{await loadStoryModel(client,manifest);if(!valid())return;
      await traceLineageProbabilities(snapshot,branch,target,manifest.config,(ids,edits)=>predictStory(client,ids,edits),(side,role,point)=>{if(valid())setTrace(old=>old?.key===key?{...old,rows:old.rows.map((track,i)=>i===side?{...track,[role]:[...track[role],point]}:track) as LineageProbabilityRows}:old);},()=>!valid());
      if(valid())setTrace(old=>old?{...old,busy:false,complete:true,status:'Lineage comparison complete.'}:old);
    }catch(e){if(valid())setTrace(old=>old?{...old,busy:false,complete:false,status:'Lineage comparison failed. Results are partial.',error:e instanceof Error?e.message:String(e)}:old);}
    finally{client.dispose();if(valid()){owner.current=null;onBusy(false);}}
  }
  function download(){if(!current)return;const data={format:'transformer-story-lineage-probabilities',version:1,checkpoint:modelKey,backend:'cpu',snapshot,childBranch:branch,targetTokenId:target,display:metric,rows,recent,cumulative,earlier,
    complete:current.complete&&rows.every((track,side)=>roles.every(role=>track[role].length===counts[side][role])),status:current.status,error:current.error||undefined,
    definitions:{original:'historical root path resolved through actual ancestry',horizon:'positions covered by the latest A/B continuations, aligned from each latest fork',recent:'latest minus immediate parent',cumulative:'latest minus original',earlier:'immediate parent minus original'}};
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download='story-lineage-probabilities.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),0);}
  const coordinates=rows.flatMap((track,side)=>[...new Set(roles.flatMap(role=>track[role].map(point=>point.offset)))].sort((a,b)=>a-b).map(offset=>`${side}:${offset}`)),tabStop=coordinates.includes(focused)?focused:coordinates[0];
  function move(event:KeyboardEvent<HTMLButtonElement>,side:number,offset:number){if(event.altKey||event.metaKey)return;let next:string|undefined;const offsets=coordinates.filter(p=>p.startsWith(`${side}:`));
    if(event.key==='Home')next=event.ctrlKey?coordinates[0]:offsets[0];else if(event.key==='End')next=event.ctrlKey?coordinates.at(-1):offsets.at(-1);else if(event.key==='ArrowLeft')next=`${side}:${offset-1}`;else if(event.key==='ArrowRight')next=`${side}:${offset+1}`;else if(event.key==='ArrowUp')next=`${side-1}:${offset}`;else if(event.key==='ArrowDown')next=`${side+1}:${offset}`;else return;
    event.preventDefault();if(next&&coordinates.includes(next)){const [s,i]=next.split(':');chart.current?.querySelector<HTMLButtonElement>(`button[data-side="${s}"][data-offset="${i}"]`)?.focus();}}
  function missing(side:number,role:LineageRole,offset:number){return offset<counts[side][role]?'Not measured':'No saved prediction at this step';}
  return <section aria-label="Original, parent and latest story probabilities">
    <p className="eyebrow">Follow repeated rebranching</p><h4>What changed recently, and what changed overall?</h4>
    <p>Original is the saved source-story path at the root of each branch’s actual ancestry. Compare it with the immediate parent and latest continuation, at matching story steps from the latest fork. Every measurement uses that version’s own preceding text, active head settings, and recorded sampling draw.</p>
    <div className="continuation-actions"><label>Latest paths<select aria-label="Lineage comparison latest paths" value={branch} disabled={disabled} onChange={event=>setBranch(event.target.value as typeof branch)}><option value="edited">With latest head changes</option><option value="original">Keep parent settings</option></select></label>
      <label>Chosen next token<select aria-label="Lineage comparison chosen token" value={target} disabled={disabled} onChange={event=>setTarget(Number(event.target.value))}>{options}</select></label>
      <label>Show<select aria-label="Lineage comparison display" value={metric} onChange={event=>setMetric(event.target.value as Metric)}><option value="probability">Original, parent, and latest probability</option><option value="recent">Latest minus parent · most recent change</option><option value="cumulative">Latest minus original · cumulative change</option></select></label>
      <button type="button" disabled={busy||disabled||!count} onClick={()=>void start()}>Measure original, parent and latest probabilities</button><button type="button" disabled={!busy} onClick={stop}>Stop lineage comparison</button><button type="button" disabled={busy||!current} onClick={download}>Download lineage comparison</button>
    </div>
    <p data-testid="lineage-target">Following <strong>{tokenLabel(tokenizer,target)}</strong> (token {target}) as the possible next token at every position. Actually generated tokens can differ; words can span model tokens.</p>
    <p role="status" data-testid="lineage-status">{current?.status??'Ready to compare the three saved versions.'} {all.length} / {total} predictions measured.</p>{busy&&<progress aria-label="Lineage probability measurement progress" value={all.length} max={total}/>}{current?.error&&<p role="alert">Could not finish: {current.error} Completed measurements are kept.</p>}
    <p className="hint">{metric==='probability'?`All six paths share 0–${percent(probabilityLimit)}. Green: original · Gray: parent · Purple: latest.`:`Both stories and both difference views share ±${(differenceLimit*100).toFixed(4)} pp. Orange: higher in latest · Blue: lower.`} pp = percentage points. Missing saved tails are unavailable; hatched bars are unmeasured. After text diverges, differences include changed context as well as head settings.</p>
    <div ref={chart} className="parent-trace-scroll lineage-scroll" role="region" aria-label="Original, parent and latest probability tracks" tabIndex={0} data-probability-limit={probabilityLimit} data-difference-limit={differenceLimit} data-metric={metric}>
      {snapshot.children.map((node,side)=><div key={node.id} className="parent-trace-track" role="group" aria-label={`Story ${side===0?'A':'B'} lineage track`}><h5>{side===0?'A':'B'} · {group.labels[side]}</h5>
        <p className="hint" data-testid={`lineage-origin-${side}`}>Original: {snapshot.originals[side].branch==='original'?'source-settings path':'path with the initial head change'} · {snapshot.originals[side].links.length} forks to latest · latest fork at story step {node.plan.source.step}.</p>
        <details className="lineage-history"><summary>How this branch traces back to its original story</summary><ol>{snapshot.originals[side].links.map((link,index)=><li key={link.nodeId}>Fork {index+1} began at story step {link.sourceStep}, from the {link.sourceBranch==='original'?'source-settings':'changed'} path.</li>)}</ol></details>
        <div className="parent-trace-columns" style={{gridTemplateColumns:`repeat(${count},minmax(120px,1fr))`,minWidth:count*120}}>{Array.from({length:count},(_,offset)=>{
          const values=roles.map(role=>({role,point:rows[side][role].find(p=>p.offset===offset)})),a=recent[side].find(p=>p.offset===offset),b=cumulative[side].find(p=>p.offset===offset),difference=metric==='cumulative'?b:a;
          const labels=values.map(({role,point})=>`${label(role)}: ${point?`${tokenLabel(tokenizer,point.source.choice.tokenId)} · ${percent(point.probability)}`:missing(side,role,offset)}`);
          const content=<><small>Offset {offset+1} · step {node.plan.source.step+offset}</small><span>{labels[0]}</span><span>{labels[1]}</span><span className={`parent-trace-plot ${metric==='probability'?'lineage-three-bars':`parent-trace-signed ${difference?'':'parent-trace-no-difference'}`}`} aria-hidden="true">{metric==='probability'?values.map(({role,point})=>point?<i key={role} className={`lineage-bar lineage-${role}`} style={{height:`${probabilityLimit?point.probability/probabilityLimit*96:0}%`}}/>:offset<counts[side][role]?<i key={role} className={`lineage-bar lineage-${role} parent-trace-pending`} style={{height:'96%'}}/>:null):difference?<i className="parent-trace-difference" style={{height:`${differenceLimit?Math.abs(difference.difference)/differenceLimit*48:0}%`,...(difference.difference>=0?{bottom:'50%'}:{top:'50%'}),background:interactionFill(difference.difference,differenceLimit)}}/>:null}</span><span>{labels[2]}</span><strong>{metric==='cumulative'?'Cumulative':'Recent'}: {difference?points(difference.difference):'No paired value'}</strong></>;
          return values.some(p=>p.point)?<button key={offset} type="button" className="parent-trace-point lineage-point" data-side={side} data-offset={offset} data-original={values[0].point?.probability} data-parent={values[1].point?.probability} data-latest={values[2].point?.probability} data-recent={a?.difference} data-cumulative={b?.difference} aria-pressed={selected===offset} disabled={busy} tabIndex={`${side}:${offset}`===tabStop?0:-1} onFocus={()=>setFocused(`${side}:${offset}`)} onKeyDown={event=>move(event,side,offset)} onClick={()=>setSelected(offset)} aria-label={`Story ${side===0?'A':'B'}, offset ${offset+1}, step ${node.plan.source.step+offset}. Target ${tokenLabel(tokenizer,target)}. ${labels.join('. ')}. Recent ${a?points(a.difference):'not available'}. Cumulative ${b?points(b.difference):'not available'}.`}>{content}</button>:<div key={offset} className="parent-trace-unmeasured">{content}</div>;
        })}</div>
      </div>)}
    </div>
    <p className="hint">Select an offset to inspect all three versions in both stories. Tab enters a measured point; arrows move between offsets and A/B, Enter selects. Offsets align generation steps, not words. Older history is measured only over the latest continuations’ horizon.</p>
    <div className="parent-trace-details">{snapshot.children.map((node,side)=>{const a=recent[side].find(p=>p.offset===selected),b=cumulative[side].find(p=>p.offset===selected),c=earlier[side].find(p=>p.offset===selected);return <article key={node.id} aria-label={`Story ${side===0?'A':'B'} lineage details`}><h5>{side===0?'A':'B'} · offset {selected+1} · story step {node.plan.source.step+selected}</h5>
      <p>Latest − parent: <strong data-testid={`lineage-recent-${side}`}>{a?points(a.difference):'No paired value'}</strong></p><p>Parent − original: <strong data-testid={`lineage-earlier-${side}`}>{c?points(c.difference):'No paired value'}</strong></p><p>Latest − original: <strong data-testid={`lineage-cumulative-${side}`}>{b?points(b.difference):'No paired value'}</strong></p>
      {a&&b&&c&&<p>Recent change plus earlier change equals the cumulative change before display rounding. These compare each version’s own text and settings.</p>}
      {roles.map(role=>{const point=rows[side][role].find(p=>p.offset===selected),parent=snapshot.parents[side],nodeId=role==='child'?node.id:role==='parent'&&parent.kind==='node'?parent.node.id:'';return <div key={role} className="parent-trace-detail" role="group" aria-label={`${side===0?'A':'B'} ${label(role).toLowerCase()} lineage prediction`}><h6>{label(role)} story</h6>{point?<><p>Chosen-token probability: <strong data-testid={`lineage-${role}-${side}`}>{percent(point.probability)}</strong></p><p>Actually generated: {tokenLabel(tokenizer,point.source.choice.tokenId)} · recorded probability {percent(point.source.choice.probability)}. Recorded choice reproduced.</p>
        <details><summary>Exact preceding text and settings</summary><p className="inspection-prefix">{encoder.preview(point.source.prefix)}</p><p>{selectionEdits(point.source).map(edit=>`Layer ${edit.layer+1} · Head ${edit.head+1} · ${edit.factor}×`).join('; ')||'No head edits.'}</p><p>Temperature {point.source.settings.temperature} · top-p {point.source.settings.topP} · seed {point.source.settings.seed} · global step {point.source.step}.</p></details>
      </>:<p>{missing(side,role,selected)}.</p>}<button type="button" disabled={busy||disabled} onClick={()=>{stop();onOpenStory(nodeId);}}>Open {side===0?'A':'B'} {label(role).toLowerCase()} story</button></div>;})}
    </article>;})}</div>
    <p className="hint">This reads saved histories without extending them. Measurements are temporary; changing target, paths, saved inputs, or closing clears them. Download the lineage comparison to retain probabilities, ancestry, exact prefixes, and completeness. The parent-only view retains its ranking and paired rebranch controls.</p>
  </section>;
}
