"""Mixed-board runs keep one tensor format, explicit device, and isolated state."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from weiqi.katago_hardware import HardwareProfile
from weiqi.katago_mixed import MixedBoardRunner, load_mixed_config
from weiqi.katago_rl_cli import build_parser, main
from weiqi.katago_rl import KataGoRLRunner, KataGoRLRunnerError
from weiqi.rl_metrics import parse_sgfs


class MixedBoardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "mixed.json"
        self.document = {
            "schema_version": 1,
            "board_sizes": [7, 8, 9, 13, 19],
            "board_weights": [2, 2, 3, 2, 1],
            "shuffle_rows_per_file": 256,
            "training": {"schema_version": 1, "preset": "balanced", "overrides": {
                "game": {"board_size": 19},
                "network": {"channels": 128, "residual_blocks": 10},
                "hardware": {"device": "cpu", "precision": "float32"},
                "runtime": {"output_directory": str(self.root / "run")},
            }},
        }
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.document))

    def runner(self):
        return MixedBoardRunner(self.path, project_root=self.root)

    def test_five_sizes_share_one_19_board_network_without_expanding_native_game(self):
        plan = load_mixed_config(self.path)
        self.assertEqual(plan.board_sizes, (7, 8, 9, 13, 19))
        self.assertEqual(plan.board_weights, (2, 2, 3, 2, 1))
        self.assertEqual(plan.training.game.board_size, 19)
        self.assertEqual(self.runner().model_kind, "b10c128")

    def test_invalid_sampling_and_tensor_geometry_are_rejected(self):
        cases = [
            ("board_sizes", [7, 7, 19]), ("board_sizes", [7, True, 19]),
            ("board_sizes", [6, 8, 19]), ("board_sizes", []),
            ("board_weights", [1]), ("board_weights", [1, 1, 1, 1, 0]),
            ("board_weights", [1, 1, 1, 1, float("nan")]),
            ("shuffle_rows_per_file", 0), ("typo", 1),
        ]
        original = deepcopy(self.document)
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.document = {**deepcopy(original), key: value}
                self.write()
                with self.assertRaises(ValueError):
                    load_mixed_config(self.path)
        self.document = deepcopy(original)
        self.document["training"]["overrides"]["game"]["board_size"] = 13
        self.write()
        with self.assertRaisesRegex(ValueError, "19|棋盘|尺寸"):
            load_mixed_config(self.path)

    def test_selfplay_preserves_csv_lists_in_config_and_uses_common_data_length(self):
        runner = self.runner()
        runner.selfplay_config.parent.mkdir(parents=True)
        runner.selfplay_config.write_text("bSizes = 19\nbSizeRelProbs = 1\ndataBoardLen = 19\n")
        with patch.object(runner, "_require_runtime"), patch.object(runner, "_run", return_value="") as run:
            runner.selfplay(games=5, visits=16)
        command = list(map(str, run.call_args.args[0]))
        cfg = Path(command[command.index("-config") + 1])
        values = dict(line.split("=", 1) for line in cfg.read_text().splitlines() if "=" in line)
        values = {k.strip():v.strip() for k,v in values.items()}
        self.assertEqual(values["bSizes"], "7,8,9,13,19")
        self.assertEqual(values["bSizeRelProbs"], "2,2,3,2,1")
        self.assertEqual(values["dataBoardLen"], "19")
        self.assertNotIn("-override-config", command)

    def test_shuffle_forwards_actual_batch_and_new_data_budget_and_keeps_small_shards(self):
        runner = self.runner()
        runner.ensure_manifest()
        path = runner.selfplay_dir / "gogogo-s1-d64/tdata/data.npz"
        path.parent.mkdir(parents=True)
        np.savez_compressed(path,
            binaryInputNCHWPacked=np.packbits(np.ones((64,22,361),dtype=np.uint8),axis=2),
            globalInputNC=np.zeros((64,1),dtype=np.float32),
            valueTargetsNCHW=np.zeros((64,5,19,19),dtype=np.float32))
        with patch.object(runner, "_require_runtime"), patch.object(runner, "_run", return_value="") as run:
            shuffled = runner.shuffle(min_rows=1, training_batch_size=8, new_data_rows=2)
        command = list(map(str,run.call_args.args[0]))
        self.assertEqual(command[command.index("-min-rows")+1], "12")
        self.assertEqual(command[command.index("-approx-rows-per-out-file")+1], "256")
        self.assertTrue((shuffled / "train").is_dir())

    def test_configured_cheap_search_survives_mixed_override_and_smoke_stays_bounded(self):
        self.document["training"]["overrides"]["search"] = {
            "simulations_per_move":32, "fast_simulations_per_move":8, "katago_cheap_search_visits":12}
        self.write()
        runner = self.runner()
        runner.selfplay_config.parent.mkdir(parents=True)
        runner.selfplay_config.write_text("dataBoardLen = 19\n")
        with patch.object(runner, "_require_runtime"), patch.object(runner, "_run", return_value="") as run:
            runner.selfplay(games=4)
            command=list(map(str,run.call_args.args[0]))
            normal=Path(command[command.index("-config")+1]).read_text()
            runner.selfplay(games=4,smoke=True)
            command=list(map(str,run.call_args.args[0]))
            smoke=Path(command[command.index("-config")+1]).read_text()
        self.assertIn("cheapSearchVisits = 12\n", normal)
        self.assertIn("cheapSearchVisits = 8\n", smoke)
        self.assertIn("maxVisits = 8\n", smoke)

    def test_targeted_positions_resolve_relative_path_before_engine_launch(self):
        starts=self.root / "targeted"
        starts.mkdir()
        (starts/"sample.startposes.txt").write_text("{}\n")
        self.document["training"]["overrides"]["self_play"] = {
            "targeted_start_probability":.1,"targeted_positions_directory":"targeted"}
        self.write()
        runner=self.runner()
        runner.ensure_manifest()
        model=runner.models_dir / "gogogo-s64-d64/model.bin.gz"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"frozen model")
        runner.selfplay_config.parent.mkdir(parents=True)
        runner.selfplay_config.write_text("dataBoardLen = 19\n")
        with patch.object(runner,"_require_runtime"),patch.object(runner,"_run",return_value="") as run:
            runner.selfplay(games=4)
        command=list(map(str,run.call_args.args[0]))
        config=Path(command[command.index("-config")+1]).read_text()
        self.assertIn(f"hintPosesDir = {starts.resolve()}\n",config)
        self.assertIn("hintPosesProb = 0.1\n",config)
        (starts/"sample.startposes.txt").unlink()
        with patch.object(runner,"_require_runtime"),patch.object(runner,"_run") as run:
            with self.assertRaisesRegex(KataGoRLRunnerError,"定向采样"):
                runner.selfplay(games=4)
        run.assert_not_called()

    def test_mps_game_threads_can_be_tuned_without_changing_search_or_batch(self):
        profile = HardwareProfile("mps", "mps", "float32", "Apple M4", 11.84, 4, False)
        baseline = MixedBoardRunner(self.path, project_root=self.root, hardware_profile=profile)
        expected = baseline.selfplay_overrides()
        self.assertEqual(expected["numGameThreads"], 2)
        self.document["mps_game_threads"] = 8
        self.write()
        tuned = MixedBoardRunner(self.path, project_root=self.root, hardware_profile=profile)
        self.assertEqual(tuned.selfplay_overrides(), {**expected, "numGameThreads": 8})
        self.assertEqual(tuned.manifest_identity(), baseline.manifest_identity())

    def test_mps_hybrid_only_changes_modeled_selfplay_inference(self):
        profile = HardwareProfile("mps", "mps", "float32", "Apple M4", 11.84, 4, False)
        baseline = MixedBoardRunner(self.path, project_root=self.root, hardware_profile=profile)
        self.document["mps_hybrid_inference"] = True
        self.write()
        hybrid = MixedBoardRunner(self.path, project_root=self.root, hardware_profile=profile)
        self.assertEqual(hybrid.selfplay_overrides(), {
            **baseline.selfplay_overrides(),
            "numNNServerThreadsPerModel": 4,
            "metalDeviceToUseThread1": 0,
            "metalDeviceToUseThread2": 100,
            "metalDeviceToUseThread3": 100,
        })
        for options in ({"has_model": False}, {"smoke": True}):
            with self.subTest(options=options):
                self.assertEqual(hybrid.selfplay_overrides(**options), baseline.selfplay_overrides(**options))
        self.assertEqual(hybrid.manifest_identity(), baseline.manifest_identity())
        self.assertNotIn("metalDeviceToUseThread2", self.runner().selfplay_overrides())

    def test_manifest_prevents_incompatible_resume_before_changing_existing_data(self):
        runner = self.runner()
        runner.ensure_manifest()
        manifest = runner.run_root / "mixed_manifest.json"
        before = manifest.read_bytes()
        self.document["board_weights"] = [1, 1, 1, 1, 1]
        self.write()
        with self.assertRaisesRegex(ValueError, "不一致|不兼容"):
            self.runner().ensure_manifest()
        self.assertEqual(manifest.read_bytes(), before)

    def test_existing_unidentified_training_data_is_not_adopted(self):
        runner = self.runner()
        path = runner.selfplay_dir / "old" / "tdata" / "old.npz"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"existing data")
        with self.assertRaisesRegex(ValueError, "清单|manifest|已有"):
            runner.ensure_manifest()
        self.assertEqual(path.read_bytes(), b"existing data")

    def test_board_statistics_validate_format_and_include_small_boards(self):
        runner = self.runner()
        runner.ensure_manifest()
        folder = runner.selfplay_dir / "训练模型" / "tdata"
        folder.mkdir(parents=True)
        masks = np.zeros((5, 22, 19, 19), dtype=np.uint8)
        for row, size in enumerate((7, 8, 9, 13, 19)):
            masks[row, 0, :size, :size] = 1
        path = folder / "mixed.npz"
        np.savez_compressed(path,
            binaryInputNCHWPacked=np.packbits(masks.reshape(5,22,361),axis=2),
            valueTargetsNCHW=np.zeros((5,5,19,19),dtype=np.float32))
        stats = runner.board_statistics()
        self.assertEqual(stats["rows_by_board"], {str(size):1 for size in (7,8,9,13,19)})
        with patch("numpy.load", side_effect=AssertionError("应复用中文路径的统计缓存")):
            cached = runner.board_statistics()
        self.assertEqual(cached["rows_by_board"], stats["rows_by_board"])
        np.savez_compressed(path, binaryInputNCHWPacked=np.zeros((1,22,11),dtype=np.uint8),
                            valueTargetsNCHW=np.zeros((1,5,9,9),dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "尺寸|张量|格式"):
            runner.board_statistics()

    def test_mps_engine_resolves_on_path_and_has_no_cuda_settings(self):
        profile = HardwareProfile("mps", "mps", "float32", "Apple M4", 11.84, 4, False)
        with patch("weiqi.katago_rl.shutil.which", return_value="/opt/homebrew/bin/katago"):
            runner = MixedBoardRunner(self.path, project_root=self.root, hardware_profile=profile)
        self.assertEqual(runner.katago_executable, Path("/opt/homebrew/bin/katago"))
        overrides = runner.selfplay_overrides(has_model=True)
        self.assertEqual(overrides["metalDeviceToUseThread0"], 0)
        self.assertFalse(any(key.startswith("cuda") for key in overrides))

    def test_mixed_entry_runs_one_finite_cycle_and_status_does_not_probe_devices(self):
        args = build_parser().parse_args(["mixed", "--mixed-config", str(self.path), "--iterations", "1"])
        self.assertEqual(args.mixed_config, self.path)
        with patch("weiqi.katago_rl_cli.MixedBoardRunner") as factory, \
                patch("weiqi.katago_rl_cli.detect_hardware", return_value=HardwareProfile(
                    "cpu", "cpu", "float32", None, 0, 1, False)):
            main(["mixed", "--mixed-config", str(self.path), "--iterations", "1"])
        factory.return_value.initialize_model.assert_not_called()
        factory.return_value.cycle.assert_called_once()
        with patch("weiqi.katago_rl_cli.MixedBoardRunner") as factory, \
                patch("weiqi.katago_rl_cli.detect_hardware", side_effect=AssertionError("status probed GPU")):
            main(["mixed-status", "--mixed-config", str(self.path)])
        factory.return_value.status.assert_called_once()

    def test_first_learner_uses_a_private_seed_after_native_random_bootstrap(self):
        runner = self.runner()
        runner.ensure_manifest()
        seed = runner.run_root / "initial_model/model.ckpt"
        seed.parent.mkdir(parents=True)
        seed.write_bytes(b"seed checkpoint")
        with patch.object(runner, "initialize_model"), patch.object(KataGoRLRunner, "cycle") as cycle:
            runner.cycle(smoke=True)
        self.assertEqual(cycle.call_args.kwargs["initial_checkpoint"], seed)

    def test_bootstrap_keeps_inference_empty_until_a_trained_model_is_exported(self):
        self._run_bootstrap_cycle(produce_trained_model=True)

    def test_bootstrap_cannot_export_the_seed_when_training_produces_no_model(self):
        self._run_bootstrap_cycle(produce_trained_model=False)

    def _run_bootstrap_cycle(self, *, produce_trained_model):
        runner = self.runner()
        runner.python_source = Path(__file__).resolve().parents[1] / "katago/source/python"
        runner.selfplay_config.parent.mkdir(parents=True, exist_ok=True)
        runner.selfplay_config.write_text("dataBoardLen = 19\n")
        stages = []

        def run_external(command, **kwargs):
            command = list(map(str, command))
            if "selfplay" in command:
                self.assertEqual(list(runner.models_dir.glob("*/model.bin.gz")), [])
                stages.append("selfplay")
            else:
                self.assertEqual(stages, ["selfplay", "train"])
                self.assertIn("gogogo-s64-d32", command[command.index("-checkpoint") + 1])
                target = Path(command[command.index("-export-dir") + 1])
                (target / "model.bin").write_bytes(b"trained inference model")
                stages.append("export")
            return ""

        def train(**kwargs):
            self.assertEqual(stages, ["selfplay"])
            seed = kwargs["initial_checkpoint"]
            self.assertEqual(seed, runner.run_root / "initial_model/model.ckpt")
            self.assertTrue(seed.is_file())
            self.assertEqual(list((runner.run_root / "torchmodels_toexport").glob("*/model.ckpt")), [])
            stages.append("train")
            if produce_trained_model:
                checkpoint = runner.run_root / "torchmodels_toexport/gogogo-s64-d32/model.ckpt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"trained checkpoint")

        with patch.object(runner, "_require_runtime"), \
                patch.object(runner, "_run", side_effect=run_external), \
                patch.object(runner, "shuffle"), patch.object(runner, "train", side_effect=train), \
                patch.object(runner.metric_store, "sync", return_value=[]), redirect_stdout(io.StringIO()):
            if produce_trained_model:
                runner.cycle(smoke=True)
            else:
                with self.assertRaisesRegex(KataGoRLRunnerError, "新模型"):
                    runner.cycle(smoke=True)
        expected = ["selfplay", "train", "export"] if produce_trained_model else ["selfplay", "train"]
        self.assertEqual(stages, expected)
        self.assertEqual([path.parent.name for path in runner.models_dir.glob("*/model.bin.gz")],
                         ["gogogo-s64-d32"] if produce_trained_model else [])

    def test_resume_recovers_completed_checkpoint_before_selfplay(self):
        import torch
        runner = self.runner()
        runner.ensure_manifest()
        checkpoint = runner.run_root / "train/gogogo/checkpoint.ckpt"
        checkpoint.parent.mkdir(parents=True)
        torch.save({"train_state": {"global_step_samples": 64, "total_num_data_rows": 32}}, checkpoint)

        def run_external(command, **kwargs):
            command = list(map(str, command))
            target = Path(command[command.index("-export-dir") + 1])
            (target / "model.bin").write_bytes(b"recovered trained model")
            return ""

        def cycle(**kwargs):
            self.assertTrue((runner.models_dir / "gogogo-s64-d32/model.bin.gz").is_file())
            self.assertNotIn("initial_checkpoint", kwargs)

        with patch.object(runner, "_require_runtime"), patch.object(runner, "_run", side_effect=run_external), \
                patch.object(KataGoRLRunner, "cycle", side_effect=cycle), redirect_stdout(io.StringIO()):
            runner.cycle(smoke=True)

    def test_existing_seed_inference_resumes_with_its_legacy_learner_checkpoint(self):
        runner = self.runner()
        runner.ensure_manifest()
        seed = runner.run_root / "torchmodels_toexport/gogogo-s0-d0/model.ckpt"
        seed.parent.mkdir(parents=True)
        seed.write_bytes(b"legacy seed checkpoint")
        model = runner.models_dir / "gogogo-s0-d0/model.bin.gz"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"legacy seed inference")
        with patch.object(runner, "_require_runtime"), patch.object(KataGoRLRunner, "cycle") as cycle, \
                redirect_stdout(io.StringIO()):
            runner.cycle(smoke=True)
        self.assertEqual(cycle.call_args.kwargs["initial_checkpoint"], seed)
        self.assertFalse((runner.run_root / "initial_model/model.ckpt").exists())

    def test_unpublished_legacy_seed_cannot_masquerade_as_a_new_trained_export(self):
        runner = self.runner()
        runner.python_source = Path(__file__).resolve().parents[1] / "katago/source/python"
        runner.ensure_manifest()
        seed = runner.run_root / "torchmodels_toexport/gogogo-s0-d0/model.ckpt"
        seed.parent.mkdir(parents=True)
        seed.write_bytes(b"legacy unpublished seed")
        with patch.object(runner, "_require_runtime"), patch.object(KataGoRLRunner, "cycle"), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(KataGoRLRunnerError, "种子|seed"):
            runner.cycle(smoke=True)
        self.assertEqual(seed.read_bytes(), b"legacy unpublished seed")

    def test_warming_actor_stays_native_random_with_trained_candidates_present(self):
        runner = self.runner()
        runner.ensure_manifest()
        candidate = runner.models_dir / "gogogo-s4096-d1000/model.bin.gz"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"immature candidate")
        actor_models = runner.run_root / "bootstrap/native_models"
        actor_models.mkdir(parents=True)
        manager = Mock()
        manager.models_dir.return_value = actor_models
        runner.selfplay_config.parent.mkdir(parents=True, exist_ok=True)
        runner.selfplay_config.write_text("dataBoardLen = 19\n")
        with patch.object(runner, "bootstrap", manager, create=True), \
                patch.object(runner, "_require_runtime"), patch.object(runner, "_run", return_value="") as run:
            runner.selfplay(games=5)
        command = list(map(str, run.call_args.args[0]))
        self.assertEqual(Path(command[command.index("-models-dir") + 1]), actor_models)
        cfg = Path(command[command.index("-config") + 1]).read_text()
        self.assertNotIn("cudaUseNHWC", cfg)

    def test_cycle_records_bootstrap_after_private_seed_and_assesses_after_retention(self):
        runner = self.runner()
        stages = []
        manager = Mock()
        manager.read_state.return_value = None
        manager.start.side_effect = lambda: stages.append("start")
        manager.after_cycle.side_effect = lambda: stages.append("assess")

        def initialize(**kwargs):
            self.assertFalse(kwargs["publish"])
            stages.append("seed")

        with patch.object(runner, "bootstrap", manager, create=True), \
                patch.object(runner, "initialize_model", side_effect=initialize), \
                patch.object(KataGoRLRunner, "cycle", side_effect=lambda **kwargs: stages.append("cycle")), \
                patch("weiqi.katago_mixed.apply_retention_plan", side_effect=lambda *args: stages.append("retention")), \
                redirect_stdout(io.StringIO()):
            runner.cycle(smoke=True)
        self.assertEqual(stages, ["seed", "start", "cycle", "retention", "assess"])

    def test_bootstrap_status_only_reads_persisted_state(self):
        runner = self.runner()
        state = {"schema_version": 1, "phase": "warming", "last_probe_samples": 0}
        manager = Mock()
        manager.read_state.return_value = state
        manager.start.side_effect = AssertionError("status started bootstrap")
        manager.after_cycle.side_effect = AssertionError("status launched probe")
        with patch.object(runner, "bootstrap", manager, create=True), redirect_stdout(io.StringIO()):
            result = runner.status()
        self.assertEqual(result.get("bootstrap"), state)

    def test_trained_candidate_without_full_checkpoint_cannot_reset_to_private_seed(self):
        runner = self.runner()
        runner.ensure_manifest()
        seed = runner.run_root / "initial_model/model.ckpt"
        seed.parent.mkdir(parents=True)
        seed.write_bytes(b"private learner seed")
        candidate = runner.models_dir / "gogogo-s4096-d1000/model.bin.gz"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"immature candidate")
        manager = Mock()
        manager.read_state.return_value = {"phase": "warming", "last_probe_samples": 4096}
        manager.start.side_effect = AssertionError("resume reset bootstrap state")
        with patch.object(runner, "bootstrap", manager, create=True), \
                patch.object(runner, "_require_runtime"), patch.object(KataGoRLRunner, "cycle"), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(KataGoRLRunnerError, "检查点|checkpoint"):
            runner.cycle(smoke=True)
        self.assertEqual(candidate.read_bytes(), b"immature candidate")

    def test_pending_trained_export_without_full_checkpoint_is_not_adopted(self):
        runner = self.runner()
        runner.ensure_manifest()
        candidate = runner.run_root / "torchmodels_toexport/gogogo-s4096-d1000/model.ckpt"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"export checkpoint without optimizer")
        manager = Mock()
        manager.read_state.return_value = None
        with patch.object(runner, "bootstrap", manager, create=True), \
                patch.object(runner, "initialize_model"), patch.object(KataGoRLRunner, "cycle"), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(KataGoRLRunnerError, "检查点|checkpoint"):
            runner.cycle(smoke=True)
        self.assertEqual(candidate.read_bytes(), b"export checkpoint without optimizer")

    def test_interrupted_positive_inference_export_still_requires_full_checkpoint(self):
        runner = self.runner()
        runner.ensure_manifest()
        candidate = runner.models_dir / "gogogo-s4096-d1000.tmp-1234/model.bin.gz"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"partially published trained candidate")
        with patch.object(runner, "initialize_model"), patch.object(KataGoRLRunner, "cycle"), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(KataGoRLRunnerError, "检查点|checkpoint"):
            runner.cycle(smoke=True)

    def test_cycle_refreshes_health_once_and_status_reads_the_persisted_report(self):
        runner = self.runner()
        report = {"status": "needs_review", "updated_at": "fixed", "by_board": {"7": {"status": "needs_review"}}}
        with patch.object(runner, "initialize_model"), patch.object(KataGoRLRunner, "cycle"), \
                patch("weiqi.katago_mixed.build_mixed_health", return_value=report) as health:
            runner.cycle(smoke=True)
        health.assert_called_once_with(runner.run_root, runner.mixed.board_sizes)
        (runner.run_root / "health_by_board.json").write_text(json.dumps(report))
        output = io.StringIO()
        with redirect_stdout(output), patch("weiqi.katago_mixed.build_mixed_health", side_effect=AssertionError("status rescanned SGFs")):
            status = runner.status()
        self.assertEqual(status["mixed_health"], report)
        self.assertIn('"mixed_health"', output.getvalue())
        self.assertIn('"updated_at": "fixed"', output.getvalue())

    def test_mixed_sgf_metrics_do_not_mark_other_sizes_as_corruption(self):
        path = self.root / "mixed.sgfs"
        path.write_text("(;GM[1]SZ[7]RE[B+1];B[aa];W[bb])\n"
                        "(;GM[1]SZ[19]RE[W+2];B[cc];W[dd])\n")
        self.assertEqual(parse_sgfs(path)["damaged_games"], 1)
        result = parse_sgfs(path, allow_mixed_boards=True)
        self.assertEqual(result["sgf_games"], 2)
        self.assertEqual(result["damaged_games"], 0)
        self.assertNotIn("board_size", result)
        self.assertEqual(result["board_sizes"], [7,19])

    def test_sgf_exploration_identity_tracks_inputs_and_blocks_mutation(self):
        start=self.root/"starts";start.mkdir()
        sgf=start/"7.sgf";sgf.write_text("(;GM[1]SZ[7];B[bb];W[dd])")
        self.document["exploration"]={"start_positions_dir":str(start),"start_probability":.1}
        self.write();runner=self.runner();runner.ensure_manifest()
        opts=runner.selfplay_overrides()
        self.assertEqual(opts["startPosesProb"],.1)
        self.assertEqual(opts["startPosesFromSgfDir"],str(start.resolve()))
        self.assertEqual(opts["startPosesLoadProb"], 1.0)
        self.assertEqual(opts["startPosesTurnWeightLambda"], 0.0)
        self.assertEqual(opts["startPosesPolicyInitAreaProp"], 0.0)
        sgf.write_text("(;GM[1]SZ[7];B[cc];W[dd])")
        with self.assertRaisesRegex(ValueError,"不一致"):
            self.runner().ensure_manifest()

    def test_exploration_rejects_empty_directory_disguised_as_sgf_file(self):
        start=self.root/"starts";start.mkdir()
        (start/"not_a_file.sgf").mkdir()
        self.document["exploration"]={"start_positions_dir":str(start),"start_probability":.1}
        self.write()
        with self.assertRaisesRegex(ValueError,"SGF"):
            load_mixed_config(self.path)

    def test_exploration_rejects_invalid_controls_before_launch(self):
        for change in ({"start_probability":.1},{"fork_probability":.6},
                       {"early_fork_probability":True},{"policy_init":1},{"unknown":1}):
            with self.subTest(change=change):
                self.document["exploration"]=change;self.write()
                with self.assertRaises(ValueError):load_mixed_config(self.path)


if __name__ == "__main__":
    unittest.main()
