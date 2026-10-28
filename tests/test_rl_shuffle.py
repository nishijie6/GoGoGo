import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from weiqi.rl_shuffle import align_training_files


class AlignedTrainingTests(unittest.TestCase):
    def test_uniform_files_keep_every_target_paired_and_preserve_watermark(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train").mkdir()
            (root / "train.json").write_text('{"range":[1000,1021]}')
            start = 0
            for index, count in enumerate((5, 7, 9)):
                rows = np.arange(start, start + count, dtype=np.float32)[:, None]
                np.savez_compressed(root / "train" / f"data{index}.npz",
                                    globalInputNC=rows, globalTargetsNC=rows * 10)
                start += count
            result = align_training_files(root, batch_size=2, samples=8)
            inputs, targets = [], []
            for path in sorted((root / "train").glob("*.npz")):
                with np.load(path) as data:
                    inputs.extend(data['globalInputNC'][:, 0])
                    targets.extend(data['globalTargetsNC'][:, 0])
                self.assertEqual(json.loads(path.with_suffix('.json').read_text())['num_rows'], 8)
            np.testing.assert_array_equal(inputs, np.arange(16))
            np.testing.assert_array_equal(targets, np.arange(16) * 10)
            self.assertEqual(result['remainder_rows'], 5)
            self.assertEqual(json.loads((root / 'train.json').read_text()), {'range': [1000, 1021]})
            self.assertEqual(len(list((root / 'unaligned-train').glob('*.npz'))), 3)

    def test_insufficient_data_does_not_publish_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'train').mkdir()
            original = root / 'train/data.npz'
            np.savez_compressed(original, globalInputNC=np.zeros((5, 1)))
            with self.assertRaisesRegex(ValueError, '不足'):
                align_training_files(root, batch_size=2, samples=8)
            self.assertTrue(original.is_file())
            self.assertFalse((root / 'unaligned-train').exists())
