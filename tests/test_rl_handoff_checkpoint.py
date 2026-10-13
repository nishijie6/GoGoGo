"""Real PyTorch serialization/optimizer continuation in isolated handoff runs."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from weiqi.rl_handoff import HandoffError, check_import, export_handoff, import_handoff


@unittest.skipIf(torch is None, "PyTorch is optional")
class CheckpointHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.run = self.project / "training_runs" / "9x9"
        self.checkpoint = self.run / "train/gogogo/checkpoint.ckpt"
        self.checkpoint.parent.mkdir(parents=True)
        source = Path(__file__).resolve().parents[1] / "config/rl_training.rtx5070ti.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        document["overrides"]["runtime"]["output_directory"] = str(self.run)
        self.profile = self.project / "training.json"
        self.profile.write_text(json.dumps(document), encoding="utf-8")
        self.model = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.inputs = torch.tensor([[0.2, -0.4, 0.5], [0.7, 0.1, -0.2]])
        self._step(self.model, self.optimizer)
        self.payload = {"model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "train_state": {"global_step_samples": 1234},
                        "config": {"test_only": True}}

    def _step(self, model, optimizer):
        optimizer.zero_grad()
        loss = model(self.inputs).square().mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    def _transfer(self, payload):
        torch.save(payload, self.checkpoint)
        bundle = self.root / "snapshot.gogogo-handoff.zip"
        export_handoff(self.project, bundle, training_config=self.profile)
        destination = self.root / "computer-b"
        import_handoff(self.project, bundle, destination)
        return destination

    def test_full_optimizer_state_survives_and_next_update_matches(self):
        destination = self._transfer(self.payload)
        report = check_import(destination, load_checkpoint=True)
        self.assertTrue(report["checkpoint_loaded"])
        self.assertEqual(report["checkpoint"]["global_step_samples"], 1234)
        restored = torch.load(destination / "run/train/gogogo/checkpoint.ckpt",
                              weights_only=True, map_location="cpu")
        model = torch.nn.Linear(3, 2)
        model.load_state_dict(restored["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.load_state_dict(restored["optimizer"])
        before_step = next(iter(optimizer.state.values()))["step"].item()
        expected = self._step(self.model, self.optimizer)
        actual = self._step(model, optimizer)
        self.assertAlmostEqual(actual, expected, places=7)
        for left, right in zip(self.model.parameters(), model.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(next(iter(optimizer.state.values()))["step"].item(), before_step + 1)

    def test_inference_only_checkpoint_is_rejected_by_load_check(self):
        payload = {key: value for key, value in self.payload.items() if key != "optimizer"}
        destination = self._transfer(payload)
        with self.assertRaises(HandoffError):
            check_import(destination, load_checkpoint=True)

    def test_nonfinite_model_weights_are_rejected_by_load_check(self):
        self.payload["model"]["weight"][0, 0] = float("nan")
        destination = self._transfer(self.payload)
        with self.assertRaises(HandoffError):
            check_import(destination, load_checkpoint=True)


if __name__ == "__main__":
    unittest.main()
