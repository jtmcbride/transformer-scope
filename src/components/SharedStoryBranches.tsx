import { useEffect, useMemo, useRef, useState } from 'react';
import { BpeTokenizer, type BpeDocument } from '../engine/bpe';
import type { HeadScaleIntervention, ModelManifest } from '../engine/contracts';
import type { StorySelection } from '../continuation';
import { firstForkDifference, forkEditLabel, type ForkDuration, type ForkPlan } from '../story-fork';
import { SharedStoryFork } from '../story-shared-fork';
import { checkpointKey, type SharedStoryGroup, type StoryNode, type StoryWorkspace } from '../story-workspace';
import { tokenLabel } from '../engine/tokenizer';
import { ModelClient } from '../workers/client';
import { loadStoryModel, predictStory } from '../workers/story-runtime';
import type { HeadPairScanRow } from '../story-head-pair-scan';
import { StoryInteractionTrace } from './StoryInteractionTrace';

export interface SharedBranchDraft { targetTokenId?: number; effects?: [HeadPairScanRow,HeadPairScanRow]; sources: [StorySelection,StorySelection]; parents: [string|null,string|null]; labels: [string,string]; edits: [HeadScaleIntervention,HeadScaleIntervention] }
interface Props { workspace: StoryWorkspace; draft?: SharedBranchDraft; manifest: ModelManifest; tokenizer: BpeDocument; disabled: boolean; onBusy: (value:boolean)=>void; onSave: (group:SharedStoryGroup,nodes:[StoryNode,StoryNode])=>void; onOpenStory:(id:string)=>void }
export function SharedStoryBranches({workspace,draft,manifest,tokenizer,disabled,onBusy,onSave,onOpenStory}:Props) {
  const [selected,setSelected]=useState(''),[duration,setDuration]=useState<ForkDuration>('continuation'),[limit,setLimit]=useState(6);
  const [busy,setBusy]=useState(false),[error,setError]=useState(''),[traceBusy,setTraceBusy]=useState(false);
  const working=busy||traceBusy;
  useEffect(()=>{onBusy(working);},[working,onBusy]);
  const owner=useRef<ModelClient|null>(null),epoch=useRef(0),flight=useRef(false);
  const groups=workspace.sharedForks??[],group=groups.find(item=>item.id===selected)??groups.at(-1);
  const nodes=group?.nodeIds.map(id=>workspace.nodes.find(node=>node.id===id)!);
  const finished=!!nodes?.every(node=>node.frame.original.finish&&node.frame.edited?.finish);
  const step=nodes?Math.max(...nodes.map(node=>node.frame.step)):0;
  const max=draft?Math.min(...draft.sources.map(source=>manifest.config.contextLength-source.prefix.length)):64;
  const budget=Math.min(limit,max),encoder=useMemo(()=>new BpeTokenizer(tokenizer),[tokenizer]);
  useEffect(()=>()=>{++epoch.current;owner.current?.dispose();onBusy(false);},[onBusy]);
  function stop(){++epoch.current;flight.current=false;owner.current?.dispose();owner.current=null;setBusy(false);}
  async function run(play:boolean,fresh=false){
    if(flight.current||disabled||traceBusy)return;
    setError('');
    let captured:SharedStoryGroup,records:[StoryNode,StoryNode],session:SharedStoryFork;
    try{
      if(fresh){
        if(!draft)return;
        if(workspace.nodes.length>498)throw new Error('This workspace needs room for two comparisons. Download it and reset the story to begin another.');
        const plans=draft.sources.map(source=>({source,edit:draft.edits[0],additionalEdits:[draft.edits[1]],duration,maxTokens:budget})) as [ForkPlan,ForkPlan];
        session=new SharedStoryFork(plans,manifest.config);
        captured={id:crypto.randomUUID(),nodeIds:[crypto.randomUUID(),crypto.randomUUID()],labels:structuredClone(draft.labels),...(draft.targetTokenId===undefined?{}:{targetTokenId:draft.targetTokenId})};
        records=session.plans.map((plan,i)=>({id:captured.nodeIds[i],parentId:draft.parents[i],plan,frame:session.frames[i]})) as [StoryNode,StoryNode];
      }else{
        if(!group||!nodes||finished)return;
        captured=structuredClone(group);records=structuredClone(nodes) as [StoryNode,StoryNode];
        session=new SharedStoryFork([records[0].plan,records[1].plan],manifest.config);session.restore([records[0].frame,records[1].frame]);
      }
    }catch(e){setError(e instanceof Error?e.message:String(e));return;}
    const generation=++epoch.current,valid=()=>generation===epoch.current;
    setSelected(captured.id);flight.current=true;setBusy(true);onSave(captured,records);
    const client=new ModelClient();owner.current=client;
    try{
      await loadStoryModel(client,manifest);if(!valid())return;
      do{
        const frames=await session.step((ids,edits)=>predictStory(client,ids,edits),()=>!valid());
        if(!valid()||!frames)return;
        records=records.map((node,i)=>({...node,frame:frames[i]})) as [StoryNode,StoryNode];onSave(captured,records);
        if(!play||session.finished)break;
        await new Promise<void>(resolve=>setTimeout(resolve,300));
      }while(valid());
    }catch(e){if(valid())setError(e instanceof Error?e.message:String(e));}
    finally{client.dispose();if(valid()){owner.current=null;flight.current=false;setBusy(false);}}
  }
  function download(){
    if(!group||!nodes)return;
    const data={format:'transformer-shared-story-comparison',version:1,checkpoint:checkpointKey(manifest),group,nodes,complete:finished,backend:'cpu'};
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
    const link=document.createElement('a');link.href=url;link.download='shared-story-comparison.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),0);
  }
  return <section className="shared-story-branches" aria-label="Shared story continuations">
    <p className="eyebrow">Try the same change in both contexts</p><h3>Branch both stories with one pair</h3>
    {draft?<><p className="shared-pair-label">{forkEditLabel({source:draft.sources[0],edit:draft.edits[0],additionalEdits:[draft.edits[1]],duration,maxTokens:budget})}</p>
      {draft.targetTokenId !== undefined && draft.effects && <div className="shared-pair-preview" role="group" aria-label="Chosen-token shared pair preview" data-target-token={draft.targetTokenId}><p><strong>Probability of {tokenLabel(tokenizer,draft.targetTokenId)}</strong> with this pair. The model still chooses the generated tokens normally.</p><dl>{draft.effects.map((row,i) => <div key={i} data-testid={`shared-target-preview-${i}`}><dt>Story {i === 0 ? 'A' : 'B'} · source → with pair</dt><dd>{(row.before*100).toFixed(4)}% → {(row.after*100).toFixed(4)}% · {row.delta>0?'+':''}{(row.delta*100).toFixed(4)} pp</dd></div>)}</dl></div>}
      <div className="continuation-actions"><label>Apply both changes<select aria-label="Shared head-change duration" value={duration} disabled={working||disabled} onChange={event=>setDuration(event.target.value as ForkDuration)}><option value="continuation">From each selected token onward</option><option value="prediction">Only each selected prediction</option></select></label>
        <label>New tokens per path<input aria-label="Shared continuation token limit" type="number" min="1" max={max} value={budget} disabled={working||disabled} onChange={event=>setLimit(Number(event.target.value))}/></label>
        <button type="button" className="primary" disabled={working||disabled||max<1} onClick={()=>void run(true,true)}>Branch both stories with this pair</button></div>
      <p className="hint">Regenerate each selected token and continue from there. Both sources receive exactly this pair and duration; each keeps its own other head settings, expiry, sampling settings, and global draws. Each reference keeps its source settings. New results are saved without changing either source story.</p>
    </>:<p>Select a pair measured in both heatmaps to create a shared continuation. Saved shared comparisons remain available below.</p>}
    {groups.length>0&&<label>Saved comparison<select aria-label="Saved shared continuation" value={group!.id} disabled={working} onChange={event=>{setSelected(event.target.value);setError('');}}>{groups.map((item,i)=><option key={item.id} value={item.id}>Shared comparison {i+1} · {item.labels[0]} / {item.labels[1]}</option>)}</select></label>}
    {group&&nodes&&<>
      <p className="shared-captured-settings">Saved changes: {forkEditLabel(nodes[0].plan)} · {nodes[0].plan.duration==='prediction'?'each selected prediction only':'from each selected token onward'} · up to {nodes[0].plan.maxTokens} new tokens per path.</p>
      <div className="continuation-actions"><button type="button" disabled={working||disabled||finished} onClick={()=>void run(false)}>Step both comparisons</button><button type="button" disabled={working||disabled||finished} onClick={()=>void run(true)}>Play both comparisons</button><button type="button" disabled={!busy} onClick={stop}>Stop shared continuations</button><button type="button" disabled={working} onClick={download}>Download shared comparison</button></div>
      <p role="status" data-testid="shared-continuation-status">{busy?'Generating four paths · one token at a time…':finished?'Shared continuations complete.':`Shared continuations paused after ${step} ${step===1?'step':'steps'}.`}</p>
      <div className="shared-story-results">{nodes.map((node,side)=>{
        const common=encoder.preview(node.plan.source.prefix),different=firstForkDifference(node.frame),letter=side===0?'A':'B';
        return <article key={node.id} aria-label={`Story ${letter} continuation comparison`}><h4>Story {letter} · {group.labels[side]}</h4><p>From step {node.plan.source.step}, before {tokenLabel(tokenizer,node.plan.source.choice.tokenId)}.</p>
          <button type="button" onClick={()=>{stop();onOpenStory(node.id);}}>Open {letter} comparison</button>
          <p className="hint">{different===null?'The paths match so far.':`First change at story step ${node.plan.source.step+different}.`}</p>
          {(['original','edited'] as const).map(branch=>{const story=node.frame[branch]!,peer=node.frame[branch==='original'?'edited':'original']!;
            return <div className="shared-story-path" key={branch} aria-label={`${letter} ${branch==='original'?'keep source settings':'with same head changes'}`}><h5>{branch==='original'?'Keep source settings':'With same head changes'}</h5><p>{story.finish?`Stopped · ${story.finish}`:`${story.choices.length} new tokens`}</p>
              <p className="story-prose"><span>{common}</span><strong>{encoder.preview(story.tokenIds).slice(common.length)}</strong></p>
              <div className="shared-branch-tokens">{story.choices.map((choice,i)=><span key={i} className={choice.tokenId!==peer.choices[i]?.tokenId?'fork-different':''} title={`Step ${node.plan.source.step+i} · probability ${(choice.probability*100).toFixed(4)}%`}><small>{node.plan.source.step+i}</small>{tokenLabel(tokenizer,choice.tokenId)}</span>)}</div>
            </div>;
          })}</article>;
      })}</div>
      <StoryInteractionTrace key={group.id} group={group} nodes={nodes as [StoryNode,StoryNode]} manifest={manifest} tokenizer={tokenizer} disabled={busy||disabled} onBusy={setTraceBusy} onOpenStory={onOpenStory} />
      <p className="hint">Highlighted tokens differ from their own reference at the same generation offset. A and B can start at different steps and sample differently. Open either comparison to inspect numbered tokens in the flow and make another head change. Shared results reopen together here; paused paths resume together.</p>
    </>}
    {error&&<p role="alert">Could not continue both stories: {error} Saved results are kept. Check the settings, or resume a paused comparison.</p>}
  </section>;
}
