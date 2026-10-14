"""Package the existing TinyStories checkpoint and independent numerical fixtures. Does not train or retune quality probes."""
import hashlib
import json
import shutil
from pathlib import Path

import torch
from export_artifact import write_probe
from reference_model import serialized_tensor
from text_probes import load_model

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.cache/text/export-48k'
OUTPUT = ROOT / 'public/models/tinystories'


def main():
    torch.set_num_threads(4)
    model, tokenizer, manifest = load_model(SOURCE)
    assert manifest['weightsSha256'] == '0011a5f8e314f63325a7231dfa60664e5bab37dbd8a27c1e5c03fab45996f294'
    for filename, key in [('weights.bin', 'weightsSha256'), ('tokenizer.json', 'tokenizerSha256')]:
        assert hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest() == manifest[key]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename in ['manifest.json', 'weights.bin', 'tokenizer.json', 'training.json']:
        shutil.copyfile(SOURCE / filename, OUTPUT / filename)
    training = json.loads((SOURCE / 'training.json').read_text())
    quality = json.loads((SOURCE / 'probes.json').read_text())
    summary = dict(modelRevision=manifest['modelRevision'], status='experimental-language-model',
                   dataset=json.loads((ROOT / 'docs/tinystories-source.json').read_text()),
                   training=training['selected'], baselines=training['baselines']['nats']['mean'],
                   originalAcceptanceGate='failed', nameCopying=dict(correct=112, count=200, rate=quality['nameRepeat']['top1Rate'], threshold=.8),
                   pronounAgreement=dict(correct=200, count=200, rate=quality['pronounAgreement']['agreementRate']),
                   fullRunDeterminism='not evaluated', heldOutAudit='not evaluated',
                   note='Deployment authorized for real-text exploration. Earlier quality-gate failures remain failures; no training, threshold changes, or revised probes are performed.')
    (OUTPUT / 'evaluation.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Short real-text sequence keeps the on-demand full gradient/Adam verification fixture bounded.
    tokens = tokenizer.encode('Once upon a time,')
    patches = [dict(id='probe-head', kind='head.scale', layer=1, head=2, factor=.37, tokens=dict(kind='all'))]
    write_probe(model, OUTPUT, tokens, patches)
    prompts = ['Once upon a time, there was a little girl named Lily. One day,', 'Tom found a red ball. He wanted to', 'Lily said, "Hello!"\nA café has 🙂.']
    cases = []
    with torch.no_grad():
        for text in prompts:
            ids = tokenizer.encode(text)
            assert len(ids) <= model.config.contextLength
            base, _ = model(torch.tensor([ids]), capture_activations=False)
            variants = []
            for layer, head, factor, scope in [(0, 0, 0, dict(kind='all')), (3, 2, 1.5, dict(kind='all')), (3, 0, 0, dict(kind='indices', indices=[0]))]:
                patch = dict(id='text-head', kind='head.scale', layer=layer, head=head, factor=factor, tokens=scope)
                changed, _ = model(torch.tensor([ids]), [patch], capture_activations=False)
                variants.append(dict(intervention=patch, logits=serialized_tensor(changed[0, -1])['values']))
            cases.append(dict(text=text, tokenIds=ids, logits=serialized_tensor(base[0, -1])['values'], variants=variants))
    fixture = dict(modelRevision=manifest['modelRevision'], weightsSha256=manifest['weightsSha256'], cases=cases)
    (ROOT / 'tests/fixtures/text-inference.json').write_text(json.dumps(fixture, separators=(',', ':'), ensure_ascii=False) + '\n')
    print(json.dumps(dict(modelRevision=manifest['modelRevision'], weightsBytes=(OUTPUT / 'weights.bin').stat().st_size,
                          probeBytes=(OUTPUT / 'probe.json').stat().st_size, baselineNextTokens=[tokenizer.display(int(torch.tensor(c['logits']).argmax())) for c in cases]), indent=2))


if __name__ == '__main__':
    main()
