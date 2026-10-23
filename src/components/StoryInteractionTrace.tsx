import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { BpeTokenizer, type BpeDocument } from '../engine/bpe';
import type { ModelManifest } from '../engine/contracts';
import { tokenLabel } from '../engine/tokenizer';
import { forkEditLabel, selectForkToken } from '../story-fork';
import { interactionTrend, traceStoryInteractions, type InteractionTracks } from '../story-interaction-trajectory';
import { checkpointKey, type SharedStoryGroup, type StoryNode } from '../story-workspace';
import { interactionFill } from '../head-pair-heatmap';
import { ModelClient } from '../workers/client';
import { loadStoryModel, withStoryPrefix } from '../workers/story-runtime';
import '../story-interaction-trace.css';

interface Props {group:SharedStoryGroup;nodes:[StoryNode,StoryNode];manifest:ModelManifest;tokenizer:BpeDocument;disabled:boolean;onBusy:(value:boolean)=>void;onOpenStory:(id:string)=>void}
type Trace = {key:string;rows:InteractionTracks;busy:boolean;status:string;error:string;complete:boolean};
const EMPTY:InteractionTracks=[[],[]];
const points=(n:number)=>{const value=Number((n*100).toFixed(4));return `${value>0?'+':''}${value.toFixed(4)} pp`;};
const probability=(n:number)=>`${(n*100).toFixed(4)}%`;
export function StoryInteractionTrace({group,nodes,manifest,tokenizer,disabled,onBusy,onOpenStory}:Props){
  const [branch,setBranch]=useState<'original'|'edited'>('edited'),[trace,setTrace]=useState<Trace|null>(null),[selected,setSelected]=useState(0),[focused,setFocused]=useState('');
  const key=JSON.stringify([checkpointKey(manifest),group,nodes,branch]);
  const captured=useMemo(()=>structuredClone({group,nodes,branch}),[key]);
  const owner=useRef<ModelClient|null>(null),epoch=useRef(0),chart=useRef<HTMLDivElement>(null);
  const current=trace?.key===key?trace:null,rows=current?.rows??EMPTY,busy=!!current?.busy;
  const counts=captured.nodes.map(node=>node.frame[branch]!.choices.length),total=counts[0]+counts[1],count=Math.max(...counts);
  const limit=rows.flat().reduce((max,row)=>Math.max(max,Math.abs(row.interaction)),0),encoder=useMemo(()=>new BpeTokenizer(tokenizer),[tokenizer]);
  useEffect(()=>{setTrace(null);setSelected(0);setFocused('');return()=>{++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);};},[key,onBusy]);
  function stop(){++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);setTrace(old=>old?.busy?{...old,busy:false,status:'Token trace stopped. Results are partial.',complete:false}:old);}
  async function start(){
    if(busy||disabled||!total)return;
    const generation=++epoch.current,valid=()=>generation===epoch.current;
    setTrace({key,rows:[[],[]],busy:true,status:'Measuring token interactions…',error:'',complete:false});setSelected(0);setFocused('');onBusy(true);
    const client=new ModelClient();owner.current=client;
    try{
      await loadStoryModel(client,manifest);if(!valid())return;
      await traceStoryInteractions([captured.nodes[0].plan,captured.nodes[1].plan],[captured.nodes[0].frame,captured.nodes[1].frame],captured.branch,manifest.config,
        (source,use)=>withStoryPrefix(client,source.prefix,use),(side,point)=>{if(valid())setTrace(old=>old?.key===key?{...old,rows:side===0?[[...old.rows[0],point],old.rows[1]]:[old.rows[0],[...old.rows[1],point]]}:old);},()=>!valid());
      if(valid())setTrace(old=>old?{...old,busy:false,status:'Token interactions complete.',complete:true}:old);
    }catch(e){if(valid())setTrace(old=>old?{...old,busy:false,status:'Token trace failed. Results are partial.',error:e instanceof Error?e.message:String(e),complete:false}:old);}
    finally{client.dispose();if(valid()){owner.current=null;onBusy(false);}}
  }
  function download(){
    if(!current)return;
    const data={format:'transformer-story-interaction-trajectory',version:1,checkpoint:checkpointKey(manifest),backend:'cpu',...captured,rows,complete:current.complete&&rows.every((track,i)=>track.length===counts[i]),status:current.status,error:current.error||undefined,formula:'interaction = p(pair) - p(head 1) - p(head 2) + p(source settings)'};
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='story-interaction-trajectory.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),0);
  }
  const coordinates=rows.flatMap((track,side)=>track.map(point=>`${side}:${point.index}`)),tabStop=coordinates.includes(focused)?focused:coordinates[0];
  function move(event:KeyboardEvent<HTMLButtonElement>,side:number,index:number){
    if(event.altKey||event.metaKey)return;
    let target:string|undefined;
    const track=rows[side];
    if(event.key==='Home')target=event.ctrlKey?coordinates[0]:`${side}:${track[0]?.index}`;
    else if(event.key==='End')target=event.ctrlKey?coordinates.at(-1):`${side}:${track.at(-1)?.index}`;
    else if(event.key==='ArrowLeft')target=`${side}:${index-1}`;
    else if(event.key==='ArrowRight')target=`${side}:${index+1}`;
    else if(event.key==='ArrowUp')target=`${side-1}:${index}`;
    else if(event.key==='ArrowDown')target=`${side+1}:${index}`;
    else return;
    event.preventDefault();if(target&&coordinates.includes(target)){const [s,i]=target.split(':');chart.current?.querySelector<HTMLButtonElement>(`button[data-track="${s}"][data-offset="${i}"]`)?.focus();}
  }
  return <section className="story-interaction-trace" aria-label="Head-pair interaction along the stories">
    <p className="eyebrow">Follow the pair through the story</p><h4>Where does the interaction grow, fade, or reverse?</h4>
    <p className="shared-pair-label">{forkEditLabel(captured.nodes[0].plan)} · {captured.nodes[0].plan.duration==='prediction'?'only the starting prediction':'throughout the continuation'}</p>
    <p>Measure the saved pair at each completed token. Each point uses that path’s own text prefix and generated token, with the source settings as the reference. Interaction is the combined probability change minus the sum of the individual head changes.</p>
    <div className="continuation-actions"><label>Paths<select aria-label="Interaction trajectory paths" value={branch} disabled={busy||disabled} onChange={event=>setBranch(event.target.value as typeof branch)}><option value="edited">With same head changes</option><option value="original">Keep source settings</option></select></label>
      <button type="button" disabled={busy||disabled||!total} onClick={()=>void start()}>Measure token-by-token interactions</button><button type="button" disabled={!busy} onClick={stop}>Stop token trace</button><button type="button" disabled={busy||!current} onClick={download}>Download token interactions</button></div>
    <p role="status" data-testid="trajectory-status">{current?.status??(total?'Ready to measure token interactions.':'Generate or resume shared story tokens to begin.')} {rows[0].length+rows[1].length} / {total} tokens measured.</p>
    {busy&&<progress value={rows[0].length+rows[1].length} max={total} aria-label="Token interaction measurement progress"/>}
    {current?.error&&<p role="alert">Could not finish this trace: {current.error} Completed points are kept.</p>}
    {total>0&&<>
      <p className="trajectory-legend">Both tracks use ±{(limit*100).toFixed(4)} pp. Blue: negative interaction · Orange: positive · pp = percentage points. Zero is the middle line; hatched points are unmeasured. “Grow,” “fade,” and “reverse” describe interaction at the displayed precision, not the total probability effect.</p>
      <div ref={chart} className="trajectory-scroll" role="region" aria-label="Scrollable token interaction tracks" tabIndex={0} data-color-limit={limit}>
        {captured.nodes.map((node,side)=>{
          const track=rows[side],peak=track.reduce<typeof track[number]|undefined>((best,row)=>!best||Math.abs(row.interaction)>Math.abs(best.interaction)?row:best,undefined);
          const reversals=track.filter((row,i)=>interactionTrend(row,track[i-1])==='Reversed').length;
          return <div className="trajectory-track" role="group" aria-label={`Story ${side===0?'A':'B'} interaction track`} key={node.id}>
            <h5>{side===0?'A':'B'} · {captured.group.labels[side]}</h5>
            <p className="hint">{peak?`Largest magnitude ${points(peak.interaction)} at offset ${peak.index+1}, story step ${peak.step}. ${reversals} sign ${reversals===1?'reversal':'reversals'} at displayed precision.`:'No measured points yet.'}</p>
            <div className="trajectory-columns" style={{gridTemplateColumns:`repeat(${count},minmax(56px,1fr))`,minWidth:count*56}}>{Array.from({length:count},(_,index)=>{
              const point=track.find(row=>row.index===index),choice=node.frame[branch]!.choices[index],previous=track.find(row=>row.index===index-1);
              const label=choice?tokenLabel(tokenizer,choice.tokenId):'No token',trend=point?(point.active?interactionTrend(point,previous):'Expired'):choice?'Not measured':'No token';
              return point?<button type="button" className="trajectory-point" key={index} data-track={side} data-offset={index} data-interaction={point.interaction} data-active={point.active} disabled={busy}
                aria-label={`Story ${side===0?'A':'B'}, offset ${index+1}, step ${point.step}, ${label}. Interaction ${points(point.interaction)}. ${trend}.`} aria-pressed={selected===index}
                tabIndex={`${side}:${index}`===tabStop?0:-1} onFocus={()=>setFocused(`${side}:${index}`)} onKeyDown={event=>move(event,side,index)} onClick={()=>setSelected(index)}
                title={`Step ${point.step} · ${label}\nInteraction ${points(point.interaction)}\nCombined probability change ${points(point.delta)}\n${trend}`}>
                <span className="trajectory-plot" aria-hidden="true"><span className="trajectory-zero-line"/><i className={point.active?'':'trajectory-expired'} style={{height:`${limit?Math.abs(point.interaction)/limit*48:0}%`,...(point.interaction>=0?{bottom:'50%'}:{top:'50%'}),background:interactionFill(point.interaction,limit)}}/>{point.interaction===0&&<b className="trajectory-zero-mark">0</b>}</span>
                <small>{index+1} · step {point.step}</small><span className="trajectory-token">{label}</span><em>{trend}</em>
              </button>:<div key={index} className={choice?'trajectory-pending':'trajectory-ended'}><span className="trajectory-plot"/><small>Offset {index+1}</small><span className="trajectory-token">{label}</span><em>{trend}</em></div>;
            })}</div>
          </div>;
        })}
      </div>
      <p className="hint">Offsets align generation steps, not words. Tab enters a measured point; arrows move between tokens or tracks, Enter selects. Each point can target a different token and context, so changes do not isolate one cause. For edits applied only once, later “Expired” zeroes reflect the saved duration, not a natural fade.</p>
      <div className="trajectory-details" aria-label="Selected trajectory token details">{captured.nodes.map((node,side)=>{
        const point=rows[side].find(row=>row.index===selected),previous=rows[side].find(row=>row.index===selected-1),letter=side===0?'A':'B';
        if(!point)return <article key={side}><h5>{letter} · offset {selected+1}</h5><p>{selected>=counts[side]?'No generated token at this offset.':'Not measured yet.'}</p></article>;
        const source=selectForkToken(node.plan,node.frame,branch,point.index);
        return <article key={side} aria-label={`Story ${letter} selected interaction`}><h5>{letter} · {tokenLabel(tokenizer,point.tokenId)} · story step {point.step}</h5>
          <strong data-testid={`trajectory-interaction-${side}`}>{points(point.interaction)}</strong><p>{point.active?interactionTrend(point,previous):'Edits expired after the selected prediction.'}{previous&&` Previous offset: ${points(previous.interaction)}; change ${points(point.interaction-previous.interaction)}.`}</p>
          <dl><div><dt>Source settings</dt><dd>{probability(point.before)}</dd></div><div><dt>First head alone</dt><dd>{probability(point.individualProbabilities[0])}</dd></div><div><dt>Second head alone</dt><dd>{probability(point.individualProbabilities[1])}</dd></div><div><dt>Both heads</dt><dd>{probability(point.after)}</dd></div><div><dt>Sum of individual changes</dt><dd>{points(point.sumDelta)}</dd></div><div><dt>Combined probability change</dt><dd>{points(point.delta)}</dd></div></dl>
          <details><summary>Exact starting text and target</summary><p className="inspection-prefix">{encoder.preview(source.prefix)}</p><p>Target token {point.tokenId} · recorded probability {probability(point.recordedProbability)}. The four measurements keep this same prefix and target.</p></details>
          <button type="button" disabled={busy} onClick={()=>{stop();onOpenStory(node.id);}}>Open {letter} comparison to inspect step {point.step}</button>
        </article>;
      })}</div>
      <p className="hint">Measure after generation completes or pauses. Resuming generation or switching saved sources clears this temporary analysis; saved stories stay intact. Download the trace to keep its full probabilities and captured inputs.</p>
    </>}
  </section>;
}
