"""Training helpers for the text checkpoint (requires PyTorch; run with npm run test:data)."""
import unittest

import numpy as np
import torch
from train_text import BOS, EOS, PAD, learning_rate, ngram_baselines, sample, window


class TrainingHelperTests(unittest.TestCase):
    tokens = np.array([10, 11, 12, 20, 21], dtype=np.uint16)
    offsets = np.array([0, 3, 5], dtype=np.int64)

    def test_windows_start_with_bos_end_with_eos_and_pad(self):
        self.assertEqual(window(self.tokens, self.offsets, 0, 0, 6), [BOS, 10, 11, 12, EOS, PAD, PAD])
        self.assertEqual(window(self.tokens, self.offsets, 0, 2, 3), [BOS, 12, EOS, PAD])
        self.assertEqual(window(self.tokens, self.offsets, 1, 0, 2), [BOS, 20, 21])

    def test_sampling_is_seeded_and_shifts_targets(self):
        first = sample(self.tokens, self.offsets, torch.Generator().manual_seed(3), 8, 4)
        second = sample(self.tokens, self.offsets, torch.Generator().manual_seed(3), 8, 4)
        for x, y in zip(first, second): self.assertTrue(torch.equal(x, y))
        x, y = first
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        self.assertTrue(bool((x[:, 0] == BOS).all()))

    def test_schedule_warms_up_then_decays_to_floor(self):
        self.assertAlmostEqual(learning_rate(0, 100, 1., 10, .1), .1)
        self.assertAlmostEqual(learning_rate(9, 100, 1., 10, .1), 1.)
        self.assertAlmostEqual(learning_rate(100, 100, 1., 10, .1), .1)

    def test_bigram_baseline_beats_unigram_on_repetitive_text(self):
        tokens = np.array([5, 6, 5, 6, 5, 6, 5, 6], dtype=np.uint16); offsets = np.array([0, 8])
        windows = dict(openings=torch.tensor([[BOS, 5, 6, 5, 6, EOS]]), randomOffsets=torch.tensor([[BOS, 5, 6, 5, PAD, PAD]]))
        result = ngram_baselines(tokens, offsets, 8, windows, smoothing=.01)['nats']['mean']
        self.assertLess(result['bigram'], result['unigram'])


if __name__ == '__main__':
    unittest.main()
