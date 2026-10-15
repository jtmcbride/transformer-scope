"""Independent historical forks using shipped weights; no training or fixture replacement."""
import json
from pathlib import Path
import torch
from text_probes import load_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    model, tokenizer, manifest = load_model(ROOT / 'public/models/tinystories')
    source = json.loads((ROOT / 'tests/fixtures/continuation-reference.json').read_text())
    cases = []
    for branch, step, layer, head, factor, duration in [
        ('original', 6, 0, 0, 0, 'continuation'),
        ('original', 6, 0, 0, 0, 'prediction'),
        ('edited', 7, 1, 1, 0, 'continuation'),
        ('edited', 7, 0, 0, 1, 'continuation'),
    ]:
        prefix = source[branch]['tokenIds'][:len(source['promptTokenIds']) + step - 1]
        selected = dict(branch=branch, step=step, prefix=prefix, choice=source[branch]['choices'][step - 1], settings=source['settings'])
        if branch == 'edited':
            selected['edit'] = source['edit']
        edit = dict(id='fork-head', kind='head.scale', layer=layer, head=head, factor=factor, tokens=dict(kind='all'))
        plan = dict(source=selected, edit=edit, duration=duration, maxTokens=12)
        inherited = [source['edit']] if branch == 'edited' else []
        changed = [item for item in inherited if (item['layer'], item['head']) != (layer, head)] + [edit]
        branches = []
        with torch.no_grad():
            for modified in (False, True):
                ids, choices = list(prefix), []
                for index in range(plan['maxTokens']):
                    edits = changed if modified and (duration == 'continuation' or index == 0) else inherited
                    logits, _ = model(torch.tensor([ids]), edits, capture_activations=False)
                    values = logits[0, -1]
                    masked = values.clone()
                    masked[0] = masked[2] = float('-inf')
                    token = int(masked.argmax())
                    choices.append(dict(tokenId=token, probability=float(values.softmax(-1)[token])))
                    ids.append(token)
                    if token == 1 or len(ids) >= model.config.contextLength:
                        break
                branches.append(dict(tokenIds=ids, choices=choices))
        case = dict(plan=plan, original=branches[0], edited=branches[1])
        cases.append(case)
        print(branch, step, duration, 'L', layer + 1, 'H', head + 1)
        for name, result in zip(('reference', 'changed'), branches):
            print(name, tokenizer.decode(result['tokenIds']))
    (ROOT / 'tests/fixtures/story-forks-reference.json').write_text(json.dumps(dict(modelRevision=manifest['modelRevision'], cases=cases), indent=2) + '\n')


if __name__ == '__main__':
    main()
