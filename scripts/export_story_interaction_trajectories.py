"""Independent token-local four-way probability measurements along shared story paths."""
import copy
import json
from pathlib import Path
import torch
from text_probes import load_model
from export_nested_story_forks import edits_at
ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    model, _, manifest = load_model(ROOT/'public/models/tinystories')
    shared = json.loads((ROOT/'tests/fixtures/story-shared-forks-reference.json').read_text())
    def measure(target=None):
        cases = []
        with torch.no_grad():
            for case in shared['cases']:
                tracks = {}
                for branch in ['original', 'edited']:
                    tracks[branch] = []
                    for plan, frame in zip(case['plans'], case['frames']):
                        rows = []
                        for index, choice in enumerate(frame[branch]['choices']):
                            prefix = frame[branch]['tokenIds'][:len(plan['source']['prefix'])+index]
                            def probability(edits):
                                logits, _ = model(torch.tensor([prefix]), edits, capture_activations=False)
                                return float(logits[0,-1].softmax(-1)[choice['tokenId'] if target is None else target])
                            before = probability(edits_at(plan,False,index))
                            after = probability(edits_at(plan,True,index))
                            singles = []
                            for edit in [plan['edit'], *plan['additionalEdits']]:
                                alone = copy.deepcopy(plan)
                                alone['edit'], alone['additionalEdits'] = edit, []
                                singles.append(probability(edits_at(alone,True,index)))
                            summed = sum(value-before for value in singles)
                            rows.append(dict(index=index,step=plan['source']['step']+index,tokenId=choice['tokenId'],recordedProbability=choice['probability'],active=plan['duration']=='continuation' or index==0,
                                             before=before,after=after,individualProbabilities=singles,delta=after-before,sumDelta=summed,interaction=(after-before)-summed))
                        tracks[branch].append(rows)
                cases.append(tracks)
        return cases
    cases = measure()
    (ROOT/'tests/fixtures/story-interaction-trajectories-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'],cases=cases),indent=2)+'\n')
    targets = [shared['cases'][0]['frames'][1]['edited']['choices'][1]['tokenId'], 1]
    fixed = [dict(tokenId=target,cases=measure(target)) for target in targets]
    (ROOT/'tests/fixtures/story-fixed-token-trajectories-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'],targets=fixed),indent=2)+'\n')
    total = sum(len(track) for case in cases for tracks in case.values() for track in tracks)
    reversals = sum(sum(a['interaction']*b['interaction']<0 for a,b in zip(track,track[1:])) for case in cases for tracks in case.values() for track in tracks)
    print(f'Exported {total} independent token-local interaction measurements across 16 paths; {reversals} raw sign changes.')


if __name__ == '__main__':
    main()
