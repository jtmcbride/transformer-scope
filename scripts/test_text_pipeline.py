"""Standard-library checks for TinyStories preparation and the byte-level BPE specification."""
import tempfile
import unittest
from pathlib import Path

from text_data import digest, stories
from text_tokenizer import BYTE_OFFSET, Tokenizer, merge_sequence, pieces, train


class StoryTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8', newline='')
        handle.write(text); handle.close()
        return Path(handle.name)

    def test_split_normalizes_and_streams_across_block_boundaries(self):
        path = self.write('fragment end.\n<|endoftext|>\nFirst story.\r\nLine two.  \n<|endoftext|>\n\n<|endoftext|>Second “story”.<|endoftext|>')
        for block in (3, 7, 1 << 20):
            self.assertEqual(list(stories(path, False, block)), ['fragment end.', 'First story.\nLine two.', 'Second “story”.'])
            self.assertEqual(list(stories(path, True, block)), ['First story.\nLine two.', 'Second “story”.'])

    def test_digest_depends_on_order_and_boundaries(self):
        self.assertNotEqual(digest(['ab', 'c']), digest(['a', 'bc']))
        self.assertNotEqual(digest(['a', 'b']), digest(['b', 'a']))


class TokenizerTests(unittest.TestCase):
    def test_pieces_cover_every_character_and_attach_one_leading_space(self):
        text = 'Hi  there, Tom!\tIt is 2023.\n“Wow” 🙂 café'
        self.assertEqual(b''.join(pieces(text)).decode('utf-8'), text)
        self.assertEqual(pieces('a  b'), [b'a', b'  ', b'b'])
        self.assertEqual(pieces(' 42'), [b' 4', b'2'])

    def test_merge_is_left_to_right_and_non_overlapping(self):
        self.assertEqual(merge_sequence([5, 5, 5], (5, 5), 9), [9, 5])
        self.assertEqual(merge_sequence([5, 5, 5, 5], (5, 5), 9), [9, 9])

    def test_training_is_deterministic_breaks_ties_by_smallest_pair_and_round_trips(self):
        texts = ['bb.aa', 'bb.aa']
        first, second = train(texts, BYTE_OFFSET + 256 + 2), train(list(texts), BYTE_OFFSET + 256 + 2)
        self.assertEqual(first, second)
        a, b = BYTE_OFFSET + ord('a'), BYTE_OFFSET + ord('b')
        # Pieces 'bb', '.', 'aa': (b, b) and (a, a) tie and (b, b) is seen first; the smaller pair wins.
        self.assertEqual(first, [(a, a), (b, b)])
        tokenizer = Tokenizer(first)
        for text in texts + ['', 'new ünïcode 🙂 text']:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.encode('aa')[0], 0)
        self.assertEqual(tokenizer.encode('aa', bos=False, eos=True)[-1], 1)

    def test_encoding_applies_lowest_rank_first(self):
        a, b, c = (BYTE_OFFSET + ord(x) for x in 'abc')
        tokenizer = Tokenizer([(b, c), (a, b)])  # rank 0 merges "bc" before "ab"
        bc = BYTE_OFFSET + 256
        self.assertEqual(tokenizer.encode('abc', bos=False), [a, bc])


if __name__ == '__main__':
    unittest.main()
