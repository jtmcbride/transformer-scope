"""Capture-free training must preserve the independent reference's values and gradients."""
import unittest
import torch
from reference_model import ReferenceTransformer


class ReferenceCaptureTests(unittest.TestCase):
    def test_capture_free_training_preserves_gradients(self):
        torch.manual_seed(7)
        model = ReferenceTransformer()
        tokens = torch.tensor([[1, 5, 9]])
        captured_logits, capture = model(tokens)
        captured_logits.sum().backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        logits, no_capture = model(tokens, capture_activations=False)
        logits.sum().backward()
        self.assertIsNone(no_capture)
        self.assertTrue(capture)
        self.assertTrue(torch.equal(logits, captured_logits))
        self.assertTrue(all(torch.equal(expected, p.grad) for expected, p in zip(gradients, model.parameters())))


if __name__ == "__main__":
    unittest.main()
