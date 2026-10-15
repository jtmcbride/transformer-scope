"""Independent greedy rollout from the shipped weights; never changes weights or quality gates."""
import json
from pathlib import Path
import torch
from text_probes import load_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    model, tokenizer, manifest = load_model(ROOT / 'public/models/tinystories')
    prompt = 'Tom found a red ball. He wanted to'
    edit = dict(id='continuation-head', kind='head.scale', layer=0, head=0, factor=0, tokens=dict(kind='all'))
    branches = []
    with torch.no_grad():
        for interventions in [[], [edit]]:
            ids = tokenizer.encode(prompt)
            choices = []
            for _ in range(16):
                logits, _ = model(torch.tensor([ids]), interventions, capture_activations=False)
                values = logits[0, -1]
                masked = values.clone()
                masked[0] = masked[2] = float('-inf')
                token = int(masked.argmax())
                choices.append(dict(tokenId=token, probability=float(values.softmax(-1)[token])))
                ids.append(token)
                if token == 1 or len(ids) >= model.config.contextLength:
                    break
            branches.append(dict(tokenIds=ids, choices=choices))
    fixture = dict(modelRevision=manifest['modelRevision'], prompt=prompt, promptTokenIds=tokenizer.encode(prompt), edit=edit,
                   settings=dict(temperature=0, topP=.9, seed=42, maxTokens=16), original=branches[0], edited=branches[1])
    (ROOT / 'tests/fixtures/continuation-reference.json').write_text(json.dumps(fixture, indent=2) + '\n')
    for name, branch in zip(['original', 'edited'], branches):
        print(name, tokenizer.decode(branch['tokenIds']))


if __name__ == '__main__':
    main()
