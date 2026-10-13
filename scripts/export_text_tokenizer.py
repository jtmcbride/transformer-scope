"""Copy the trained tokenizer into test fixtures and export Python encodings for TypeScript parity.

Writes tests/fixtures/text-tokenizer.json and tests/fixtures/bpe-parity.json (150 validation stories plus edge cases).
With --all, also writes .cache/text/validation-ids.json for `npm run parity:tokenizer` (every validation story).
"""
import json
import sys
from pathlib import Path

from text_data import OUTPUT, load as load_stories
from text_tokenizer import load

root = Path(__file__).resolve().parents[1]
tokenizer = load(OUTPUT / 'bpe.json')
(root / 'tests/fixtures/text-tokenizer.json').write_bytes(tokenizer.serialized())
edge = ['', ' ', 'Hello', '  two  spaces  ', 'tabs\tand\nnew\r\nlines', 'Numbers 2023 and 3.14!', 'Emoji 🙂 and 👩🏽‍🚀 here',
        'Café naïve façade', 'Mixed “quotes” and ‘single’ — dash…', 'ALL CAPS', 'a non-breaking space', 'Ünïcödé ⁂ 漢字', '​zero width']
stories = load_stories('validation')
cases = [dict(text=text, ids=tokenizer.encode(text)) for text in edge + stories[:150]]
for case in cases: assert tokenizer.decode(case['ids']) == case['text']
(root / 'tests/fixtures/bpe-parity.json').write_text(json.dumps(dict(cases=cases), ensure_ascii=False) + '\n')
print(f'Exported {len(cases)} parity cases.')
if '--all' in sys.argv:
    (OUTPUT / 'validation-ids.json').write_text(json.dumps([dict(text=s, ids=tokenizer.encode(s)) for s in stories], ensure_ascii=False))
    print(f'Exported all {len(stories)} validation stories.')
