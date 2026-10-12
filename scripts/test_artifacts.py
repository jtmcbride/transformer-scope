"""Corruption gates for the deployment auditor; no PyTorch dependency."""
import hashlib
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from check_artifacts import audit

SOURCE = Path(__file__).resolve().parents[1] / 'public/models/verification'


class ArtifactTests(unittest.TestCase):
    def test_weight_corruption_and_stale_revision_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'verification'
            shutil.copytree(SOURCE, output)
            audit(output)
            weights = bytearray((output / 'weights.bin').read_bytes())
            weights[0] ^= 1
            (output / 'weights.bin').write_bytes(weights)
            with self.assertRaisesRegex(AssertionError, 'weights checksum'):
                audit(output)
            shutil.copyfile(SOURCE / 'weights.bin', output / 'weights.bin')
            manifest = json.loads((output / 'manifest.json').read_text())
            manifest['provenance']['seed'] += 1
            (output / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(AssertionError, 'revision mismatch'):
                audit(output)

    def test_valid_checksums_do_not_admit_nonfinite_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'verification'
            shutil.copytree(SOURCE, output)
            weights = bytearray((output / 'weights.bin').read_bytes())
            weights[:4] = struct.pack('<f', float('nan'))
            (output / 'weights.bin').write_bytes(weights)
            manifest = json.loads((output / 'manifest.json').read_text())
            manifest['weightsSha256'] = hashlib.sha256(weights).hexdigest()
            manifest.pop('modelRevision')
            manifest['modelRevision'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            (output / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(AssertionError, 'non-finite weight'):
                audit(output)


if __name__ == '__main__':
    unittest.main()
