import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { BpeTokenizer, type BpeDocument } from '../engine/bpe';
import type { ModelManifest } from '../engine/contracts';
import { tokenLabel } from '../engine/tokenizer';
import { forkEditLabel, selectForkToken } from '../story-fork';
import { interactionTrend, traceStoryInteractions, type InteractionTracks, type InteractionTarget } from '../story-interaction-trajectory';
import { checkpointKey, type SharedStoryGroup, type StoryNode } from '../story-workspace';
import { interactionFill } from '../head-pair-heatmap';
import { ModelClient } from '../workers/client';
import { loadStoryModel, withStoryPrefix } from '../workers/story-runtime';
import '../story-interaction-trace.css';
import { useStoryWorkspace } from './StoryWorkspace';

interface Props {group:SharedStoryGroup;nodes:[StoryNode,StoryNode];manifest:ModelManifest;tokenizer:BpeDocument;disabled:boolean;onBusy:(value:boolean)=>void;onOpenStory:(id:string)=>void}
type Trace = {key:string;rows:InteractionTracks;busy:boolean;status:string;error:string;complete:boolean};
const EMPTY:InteractionTracks=[[],[]];
const points=(n:number)=>{const value=Number((n*100).toFixed(4));return `${value>0?'+':''}${value.toFixed(4)} pp`;};
const probability=(n:number)=>`${(n*100).toFixed(4)}%`;
export function StoryInteractionTrace({group,nodes,manifest,tokenizer,disabled,onBusy,onOpenStory}:Props){
  const workspace = useStoryWorkspace();
  const [branch,setBranch]=useState<'original'|'edited'>('edited'),[trace,setTrace]=useState<Trace|null>(null),[selected,setSelected]=useState(0),[focused,setFocused]=useState('');
  const [mode,setMode]=useState<'generated'|'fixed'>(group.targetTokenId===undefined?'generated':'fixed'),[chosenToken,setChosenToken]=useState(()=>group.targetTokenId??nodes[0].frame.edited?.choices[0]?.tokenId??nodes[0].plan.source.choice.tokenId);
  const [metric,setMetric]=useState<'interaction'|'probability'|'effect'>(group.targetTokenId===undefined?'interaction':'effect');
  const target:InteractionTarget=mode==='fixed'?{kind:'fixed',tokenId:chosenToken}:{kind:'generated'};
  const key=JSON.stringify([checkpointKey(manifest),group,nodes,branch,target]);
  const captured=useMemo(()=>structuredClone({group,nodes,branch,target}),[key]);
  const owner=useRef<ModelClient|null>(null),epoch=useRef(0),chart=useRef<HTMLDivElement>(null);
  const current=!disabled&&trace?.key===key?trace:null,rows=current?.rows??EMPTY,busy=!!current?.busy;
  const counts=captured.nodes.map(node=>node.frame[branch]!.choices.length),total=counts[0]+counts[1],count=Math.max(...counts);
  const limit=rows.flat().reduce((max,row)=>Math.max(max,Math.abs(row.interaction)),0),encoder=useMemo(()=>new BpeTokenizer(tokenizer),[tokenizer]);
  const effectLimit=rows.flat().reduce((max,row)=>Math.max(max,Math.abs(row.delta)),0);
  const valueField=metric==='effect'?'delta':'interaction',signedLimit=metric==='effect'?effectLimit:limit;
  const probabilityLimit=rows.flat().reduce((max,row)=>Math.max(max,row.before,row.after),0);
  const storyTokens=useMemo(()=>[...new Set(nodes.flatMap(node=>[...node.frame.original.tokenIds,...(node.frame.edited?.tokenIds??[])]).filter(id=>id!==0&&id!==2))],[nodes]);
  function pin(tokenId:number){setChosenToken(tokenId);setMode('fixed');setMetric('effect');}
  useEffect(()=>{setTrace(null);setSelected(0);setFocused('');return()=>{++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);};},[key,onBusy]);
  useEffect(()=>{if(disabled){++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);setTrace(null);setSelected(0);setFocused('');}},[disabled,onBusy]);
  function stop(){++epoch.current;owner.current?.dispose();owner.current=null;onBusy(false);setTrace(old=>old?.busy?{...old,busy:false,status:'Token trace stopped. Results are partial.',complete:false}:old);}
  async function start(){
    if(busy||disabled||!total)return;
    const generation=++epoch.current,valid=()=>generation===epoch.current;
    setTrace({key,rows:[[],[]],busy:true,status:'Measuring token interactions…',error:'',complete:false});setSelected(0);setFocused('');onBusy(true);
    const client=new ModelClient();owner.current=client;
    try{
      await loadStoryModel(client,manifest);if(!valid())return;
      await traceStoryInteractions([captured.nodes[0].plan,captured.nodes[1].plan],[captured.nodes[0].frame,captured.nodes[1].frame],captured.branch,manifest.config,
        (source,use)=>withStoryPrefix(client,source.prefix,use),(side,point)=>{if(valid())setTrace(old=>old?.key===key?{...old,rows:side===0?[[...old.rows[0],point],old.rows[1]]:[old.rows[0],[...old.rows[1],point]]}:old);},()=>!valid(),captured.target);
      if(valid())setTrace(old=>old?{...old,busy:false,status:'Token interactions complete.',complete:true}:old);
    }catch(e){if(valid())setTrace(old=>old?{...old,busy:false,status:'Token trace failed. Results are partial.',error:e instanceof Error?e.message:String(e),complete:false}:old);}
    finally{client.dispose();if(valid()){owner.current=null;onBusy(false);}}
  }
  function download(){
    if(!current)return;
    const data={format:'transformer-story-interaction-trajectory',version:2,checkpoint:checkpointKey(manifest),backend:'cpu',...captured,rows,complete:current.complete&&rows.every((track,i)=>track.length===counts[i]),status:current.status,error:current.error||undefined,display:metric,effectDefinition:'delta = p(pair) - p(source settings), on the same prefix and target',formula:'interaction = p(pair) - p(head 1) - p(head 2) + p(source settings)'};
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
    <p className="eyebrow">Follow the pair through the story</p><h4>{metric==='effect'?'Where does the pair boost or suppress the chosen token?':'Where does the interaction grow, fade, or reverse?'}</h4>
    <p className="shared-pair-label">{forkEditLabel(captured.nodes[0].plan)} · {captured.nodes[0].plan.duration==='prediction'?'only the starting prediction':'throughout the continuation'}</p>
    <p>Measure the saved pair at each completed token. Each point uses that path’s own text prefix and your selected target, with the source settings as the reference. Probability effect is the change from source settings to both heads on the same prefix. Positive values boost the target; negative values suppress it. Interaction measures how much that effect differs from the sum of the individual head changes.</p>
    <div className="continuation-actions"><label>Paths<select aria-label="Interaction trajectory paths" value={branch} disabled={busy||disabled} onChange={event=>setBranch(event.target.value as typeof branch)}><option value="edited">With same head changes</option><option value="original">Keep source settings</option></select></label>
      <label>Follow<select aria-label="Interaction trajectory target mode" value={mode} disabled={disabled} onChange={event=>{setMode(event.target.value as typeof mode);if(event.target.value==='fixed')setMetric('effect');}}><option value="generated">Generated token</option><option value="fixed">Chosen token</option></select></label>
      {mode==='fixed'&&<label>Chosen next token<select aria-label="Chosen next token" value={chosenToken} disabled={disabled} onChange={event=>pin(Number(event.target.value))}>
        {!storyTokens.includes(chosenToken)&&<option value={chosenToken}>{tokenLabel(tokenizer,chosenToken)} · token {chosenToken}</option>}
        <optgroup label="Tokens in these stories">{storyTokens.map(id=><option key={id} value={id}>{tokenLabel(tokenizer,id)} · token {id}</option>)}</optgroup>
      </select></label>}
      <label>Show<select aria-label="Interaction trajectory display" value={metric} onChange={event=>setMetric(event.target.value as typeof metric)}><option value="effect">Probability change</option><option value="interaction">Pair interaction</option><option value="probability">Token probability</option></select></label>
      <button type="button" disabled={busy||disabled||!total} onClick={()=>void start()}>Measure token-by-token interactions</button><button type="button" disabled={!busy} onClick={stop}>Stop token trace</button><button type="button" disabled={busy||!current} onClick={download}>Download token interactions</button></div>
    <p data-testid="trajectory-target">{mode==='fixed'?<>Following <strong>{tokenLabel(tokenizer,chosenToken)}</strong> (token {chosenToken}) as the next-token target at every step in both stories.</>:'Following the token actually generated at each step.'} Choose a token from the stories, or measure first and pin a prediction alternative below. Words can span several model tokens; ␠ marks a space.</p>
    <p role="status" data-testid="trajectory-status">{current?.status??(total?'Ready to measure token interactions.':'Generate or resume shared story tokens to begin.')} {rows[0].length+rows[1].length} / {total} tokens measured.</p>
    {busy&&<progress value={rows[0].length+rows[1].length} max={total} aria-label="Token interaction measurement progress"/>}
    {current?.error&&<p role="alert">Could not finish this trace: {current.error} Completed points are kept.</p>}
    {total>0&&<>
      <p className="trajectory-legend">{metric!=='probability'?<>Both tracks use ±{(signedLimit*100).toFixed(4)} pp. Blue: {metric==='effect'?'suppression':'negative interaction'} · Orange: {metric==='effect'?'boost':'positive interaction'} · pp = percentage points. Zero is the middle line.</>:<>Both tracks use 0–{(probabilityLimit*100).toFixed(4)}%. Gray: source settings · Purple: both heads. Bars compare the same target on the same prefix.</>} Hatched points are unmeasured. “Grow,” “fade,” and “reverse” describe {metric==='effect'?'the magnitude and sign of the probability change':'interaction'} at the displayed precision.</p>
      <div ref={chart} className="trajectory-scroll" role="region" aria-label="Scrollable token interaction tracks" tabIndex={0} data-color-limit={limit} data-probability-limit={probabilityLimit} data-effect-limit={effectLimit} data-metric={metric}>
        {captured.nodes.map((node,side)=>{
          const track=rows[side],peak=track.reduce<typeof track[number]|undefined>((best,row)=>!best||Math.abs(row[valueField])>Math.abs(best[valueField])?row:best,undefined);
          const reversals=track.filter((row,i)=>interactionTrend(row,track[i-1],valueField)==='Reversed').length;
          return <div className="trajectory-track" role="group" aria-label={`Story ${side===0?'A':'B'} interaction track`} key={node.id}>
            <h5>{side===0?'A':'B'} · {captured.group.labels[side]}</h5>
            <p className="hint">{peak?`Largest magnitude ${points(peak[valueField])} at offset ${peak.index+1}, story step ${peak.step}. ${reversals} sign ${reversals===1?'reversal':'reversals'} at displayed precision.`:'No measured points yet.'}</p>
            <div className="trajectory-columns" style={{gridTemplateColumns:`repeat(${count},minmax(56px,1fr))`,minWidth:count*56}}>{Array.from({length:count},(_,index)=>{
              const point=track.find(row=>row.index===index),choice=node.frame[branch]!.choices[index],previous=track.find(row=>row.index===index-1);
              const label=choice?tokenLabel(tokenizer,choice.tokenId):'No token',trend=point?(point.active?interactionTrend(point,previous,valueField):'Expired'):choice?'Not measured':'No token';
              return point?<button type="button" className="trajectory-point" key={index} data-track={side} data-offset={index} data-interaction={point.interaction} data-delta={point.delta} data-active={point.active} data-target-token={point.targetTokenId} data-before={point.before} data-after={point.after} disabled={busy}
                aria-label={`Story ${side===0?'A':'B'}, offset ${index+1}, step ${point.step}, ${label}. Target ${tokenLabel(tokenizer,point.targetTokenId)}. Source probability ${probability(point.before)}, both heads ${probability(point.after)}. Probability change ${points(point.delta)}. Interaction ${points(point.interaction)}. ${trend}.`} aria-pressed={selected===index}
                tabIndex={`${side}:${index}`===tabStop?0:-1} onFocus={()=>setFocused(`${side}:${index}`)} onKeyDown={event=>move(event,side,index)} onClick={()=>setSelected(index)}
                title={`Step ${point.step} · ${label}\nInteraction ${points(point.interaction)}\nSource ${probability(point.before)} → pair ${probability(point.after)}\nCombined probability change ${points(point.delta)}\n${trend}`}>
                {metric!=='probability'?<span className="trajectory-plot" aria-hidden="true"><span className="trajectory-zero-line"/><i className={point.active?'':'trajectory-expired'} style={{height:`${signedLimit?Math.abs(point[valueField])/signedLimit*48:0}%`,...(point[valueField]>=0?{bottom:'50%'}:{top:'50%'}),background:interactionFill(point[valueField],signedLimit)}}/>{point[valueField]===0&&<b className="trajectory-zero-mark">0</b>}</span>:
                  <span className="trajectory-plot trajectory-probability" aria-hidden="true"><i className="trajectory-source-bar" style={{height:`${probabilityLimit?point.before/probabilityLimit*96:0}%`}}/><i className="trajectory-pair-bar" style={{height:`${probabilityLimit?point.after/probabilityLimit*96:0}%`}}/>{point.before===0&&point.after===0&&<b className="trajectory-zero-mark">0</b>}</span>}
                <small>{index+1} · step {point.step}</small><span className="trajectory-token">{label}</span><em>{trend}</em>{metric==='effect'&&<span className="trajectory-effect-value">{points(point.delta)}</span>}
              </button>:<div key={index} className={choice?'trajectory-pending':'trajectory-ended'}><span className="trajectory-plot"/><small>Offset {index+1}</small><span className="trajectory-token">{label}</span><em>{trend}</em></div>;
            })}</div>
          </div>;
        })}
      </div>
      <p className="hint">Offsets align generation steps, not words. Tab enters a measured point; arrows move between tokens or tracks, Enter selects. {mode==='fixed'?'The chosen token stays fixed; the surrounding text can still differ, so changes do not isolate one cause.':'Each point can target a different token and context, so changes do not isolate one cause.'} For edits applied only once, later “Expired” zeroes reflect the saved duration, not a natural fade.</p>
      {mode==='fixed'&&rows[0][selected]&&rows[1][selected]&&<p className="trajectory-gap" data-testid="trajectory-effect-gap">At offset {selected+1}, B’s probability change minus A’s: <strong>{points(rows[1][selected].delta-rows[0][selected].delta)}</strong>. Each effect uses its own story’s prefix and source settings.</p>}
      <div className="trajectory-details" aria-label="Selected trajectory token details">{captured.nodes.map((node,side)=>{
        const point=rows[side].find(row=>row.index===selected),previous=rows[side].find(row=>row.index===selected-1),letter=side===0?'A':'B';
        if(!point)return <article key={side}><h5>{letter} · offset {selected+1}</h5><p>{selected>=counts[side]?'No generated token at this offset.':'Not measured yet.'}</p></article>;
        const source=selectForkToken(node.plan,node.frame,branch,point.index);
        return <article key={side} aria-label={`Story ${letter} selected interaction`}><h5>{letter} · {tokenLabel(tokenizer,point.targetTokenId)} · story step {point.step}</h5>
          <p data-testid={`trajectory-generated-${side}`}>Actually generated: {tokenLabel(tokenizer,point.tokenId)} · recorded probability {probability(point.recordedProbability)}.</p>
          <p>{metric==='effect'?'Probability change':'Pair interaction'}</p><strong data-testid={metric==='effect'?`trajectory-effect-${side}`:`trajectory-interaction-${side}`}>{points(point[valueField])}</strong><p>{point.active?interactionTrend(point,previous,valueField):'Edits expired after the selected prediction.'}{previous&&` Previous offset: ${points(previous[valueField])}; change ${points(point[valueField]-previous[valueField])}.`}</p>
          <dl><div><dt>Source settings</dt><dd data-testid={`trajectory-before-${side}`}>{probability(point.before)}</dd></div><div><dt>First head alone</dt><dd>{probability(point.individualProbabilities[0])}</dd></div><div><dt>Second head alone</dt><dd>{probability(point.individualProbabilities[1])}</dd></div><div><dt>Both heads</dt><dd data-testid={`trajectory-after-${side}`}>{probability(point.after)}</dd></div><div><dt>Sum of individual changes</dt><dd>{points(point.sumDelta)}</dd></div><div><dt>Combined probability change</dt><dd data-testid={`trajectory-delta-${side}`}>{points(point.delta)}</dd></div>{metric==='effect'&&<div><dt>Pair interaction</dt><dd data-testid={`trajectory-interaction-${side}`}>{points(point.interaction)}</dd></div>}</dl>
          <details><summary>Exact starting text and target</summary><p className="inspection-prefix">{encoder.preview(source.prefix)}</p><p>Measured target: {tokenLabel(tokenizer,point.targetTokenId)} (token {point.targetTokenId}). Actual generated token: {tokenLabel(tokenizer,point.tokenId)} (token {point.tokenId}). The four measurements keep this same prefix and target.</p></details>
          <div className="trajectory-candidates"><p>Choose a target from this actual prediction:</p>
            <button type="button" disabled={busy} onClick={()=>pin(point.tokenId)}>Track generated token {tokenLabel(tokenizer,point.tokenId)} from {letter}</button>
            <ul aria-label={`Story ${letter} prediction alternatives`}>{point.alternatives.map(candidate=><li key={candidate.tokenId}><button type="button" disabled={busy} onClick={()=>pin(candidate.tokenId)} aria-label={`Track alternative ${tokenLabel(tokenizer,candidate.tokenId)}, token ${candidate.tokenId} from ${letter}`}>{tokenLabel(tokenizer,candidate.tokenId)} · {probability(candidate.probability)}</button></li>)}</ul>
          </div>
          <button type="button" disabled={busy || disabled} onClick={()=>{stop();workspace?.openPairScan({nodeId:node.id,branch,index:point.index,targetTokenId:point.targetTokenId});}}>Scan head pairs for {letter}’s chosen token</button>
          <button type="button" disabled={busy} onClick={()=>{stop();onOpenStory(node.id);}}>Open {letter} comparison to inspect step {point.step}</button>
        </article>;
      })}</div>
      <p className="hint">Measure after generation completes or pauses. Resuming generation or switching targets or saved sources clears this temporary analysis; saved stories stay intact. Download the trace to keep its full probabilities and captured inputs.</p>
    </>}
  </section>;
}
