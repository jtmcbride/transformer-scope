"""Dataset invariants and leakage controls; no training or fixture regeneration."""
import unittest
from collections import Counter, defaultdict

from retrieval_data import examples, partition, SYMBOLS, dataset_digest


class RetrievalDataTests(unittest.TestCase):
    def test_targets_bind_to_unique_nonfinal_keys_and_splits_are_disjoint(self):
        splits = defaultdict(set)
        contexts = defaultdict(Counter)
        items = list(examples())
        self.assertEqual(len(items), 3072)
        for item in items:
            bindings = [(item.tokens[i], item.tokens[i + 2]) for i in range(1, 13, 4)]
            self.assertEqual(len(set(key for key, _ in bindings)), 3)
            query = item.tokens[-2]
            self.assertNotEqual(query, bindings[-1][0])
            self.assertEqual(dict(bindings)[query], item.target)
            self.assertEqual(len(item.tokens), 16)
            self.assertTrue(all(0 <= token < len(SYMBOLS) for token in item.tokens))
            splits[partition(item)].add(item.tokens)
            contexts[item.tokens[-7:]][item.target] += 1
        self.assertEqual(sum(map(len, splits.values())), 3072)
        self.assertTrue(splits['train'].isdisjoint(splits['validation']))
        self.assertTrue(splits['train'].isdisjoint(splits['test']))
        self.assertTrue(splits['validation'].isdisjoint(splits['test']))
        # For every identical local context, all four target values occur equally often.
        for count in contexts.values():
            self.assertEqual(set(count), {5, 6, 7, 8})
            self.assertEqual(len(set(count.values())), 1)
        self.assertEqual(dataset_digest(items), dataset_digest(list(examples())))

    def test_lengths_are_disjoint(self):
        self.assertEqual({len(item.tokens) for item in examples(2)}, {12})
        self.assertEqual({len(item.tokens) for item in examples(4)}, {20})


if __name__ == '__main__':
    unittest.main()
