"""Native forced exploration and independently stored policy targets."""
import unittest
import importlib.util
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('numpy') is not None
if AVAILABLE:
    import numpy as np
    from weiqi.rl.search import Node, prune_policy_visits, search
    from weiqi.rl.state import Position
from weiqi.rl_config import RLConfigError, resolve_rl_training_config
from weiqi.katago_rl import build_selfplay_overrides, KataGoRLRunner, KataGoRLRunnerError


@unittest.skipUnless(AVAILABLE, 'Optional training dependency NumPy is not installed')
class ForcedPlayoutTests(unittest.TestCase):
    def test_joseki_selfplay_plays_raw_policy_and_stores_pruned_targets(self):
        from weiqi.rl.selfplay import GameJob, play_game
        config = resolve_rl_training_config(overrides={
            'search': {'forced_playouts_coefficient': 2, 'policy_target_pruning': True,
                       'full_search_probability': 1.0, 'temperature_moves': 0},
            'self_play': {'joseki_fraction': 1.0},
        })
        def controlled_search(position, _evaluate, search_config, _rng, *, target_policy_out, **_options):
            self.assertEqual(search_config.forced_playouts_coefficient, 2)
            played = np.zeros(len(position.legal), np.float32)
            played[-1] = 1
            target_policy_out.fill(0)
            target_policy_out[np.flatnonzero(position.legal[:-1])[0]] = 1
            return played, 0.0
        with patch('weiqi.rl.selfplay.search', side_effect=controlled_search):
            result = play_game(config, GameJob(0, 1), lambda *_: None, training=True)
        self.assertIsNotNone(result.opening_id)
        self.assertGreater(result.opening_moves, 0)
        self.assertEqual(len(result.moves), result.opening_moves + 2)
        self.assertEqual(len(result.examples), 2)
        self.assertEqual([move[1] for move in result.moves[-2:]], [81, 81])
        self.assertTrue(all(example[1][-1] == 0 for example in result.examples))

    def test_targeted_path_resolves_from_project_and_missing_pool_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = Path(__file__).resolve().parents[1] / 'config/rl_training.rtx5070ti.curriculum.19x19.json'
            runner = KataGoRLRunner(config_path, project_root=root, run_root_override=root / 'run')
            runner.config = replace(runner.config, self_play=replace(runner.config.self_play,
                targeted_start_probability=.1, targeted_positions_directory='positions'))
            model = runner.models_dir / 'frozen/model.bin.gz'
            model.parent.mkdir(parents=True)
            model.write_bytes(b'not invoked')
            with patch.object(runner, '_require_runtime'):
                with self.assertRaises(KataGoRLRunnerError):
                    runner.selfplay(games=1)
                (root / 'positions').mkdir()
                (root / 'positions/test.startposes.txt').write_text('{}\n', encoding='utf8')
                with patch.object(runner, '_run', return_value='') as command:
                    runner.selfplay(games=1)
                self.assertIn(f'hintPosesDir={root / "positions"}', command.call_args.args[0][-1])

    def test_pruning_preserves_best_and_does_not_change_search_statistics(self):
        root = Node()
        root.visits = 100
        root.priors = np.array([.8, .1, .1])
        root.child_visits = np.array([80., 10., 10.])
        root.child_values = np.array([-64., 10., 10.])
        target = prune_policy_visits(root, 1.5, 2)
        self.assertEqual(target[0], 80)
        self.assertTrue((target[1:] < 10).all())
        np.testing.assert_array_equal(root.child_visits, [80, 10, 10])
        self.assertTrue((target >= 0).all())

    def test_forced_search_prunes_only_training_targets_and_uses_exact_budget(self):
        state = Position.new(9, 6.5)
        config = replace(resolve_rl_training_config().search,
                         forced_playouts_coefficient=2, policy_target_pruning=True)
        def evaluate(x):
            generator = np.random.default_rng(int(abs(x.sum() * 1000)) % 9973)
            return generator.normal(size=82), float(generator.uniform(-.9, .9))
        target = np.empty(82, np.float32)
        info = {}
        policy, value = search(state, evaluate, config, np.random.default_rng(9),
                               simulations=96, add_noise=True,
                               target_policy_out=target, diagnostics=info)
        self.assertEqual(info['visits'], 96)
        self.assertGreater(info['forced_playouts'], 0)
        self.assertGreater(info['pruned_visits'], 0)
        self.assertAlmostEqual(float(target.sum()), 1, places=6)
        self.assertTrue(np.isfinite(value))
        self.assertFalse(np.array_equal(policy, target))
        plain = replace(config, forced_playouts_coefficient=0, policy_target_pruning=False)
        a = search(state, evaluate, plain, np.random.default_rng(2), simulations=32)
        b = search(state, evaluate, config, np.random.default_rng(2), simulations=32,
                   target_policy_out=target, diagnostics=info)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(target, b[0])
        self.assertEqual(a[1], b[1])
        self.assertEqual(info['forced_playouts'], 0)

    def test_virtual_visits_do_not_leak_under_forced_batched_search(self):
        config = replace(resolve_rl_training_config().search, leaf_batch_size=4,
                         forced_playouts_coefficient=2, policy_target_pruning=True)
        def evaluate(x):
            return np.zeros(82), .5
        def batch(xs):
            return np.zeros((len(xs), 82)), np.full(len(xs), .5)
        for budget in (1, 7, 33):
            info = {}
            policy, value = search(Position.new(9, 6.5), evaluate, config,
                np.random.default_rng(1), simulations=budget, add_noise=True,
                evaluate_batch=batch, diagnostics=info)
            self.assertEqual(info['visits'], budget)
            self.assertAlmostEqual(float(policy.sum()), 1, places=6)
            self.assertLessEqual(abs(value), 1)

    def test_new_options_validate_and_targeted_starts_preserve_rules(self):
        for overrides in ({'search': {'forced_playouts_coefficient': -1}},
                          {'search': {'katago_forced_playouts_coefficient': 101}},
                          {'search': {'policy_target_pruning': 1}},
                          {'self_play': {'targeted_start_probability': .1}}):
            with self.assertRaises(RLConfigError):
                resolve_rl_training_config(overrides=overrides)
        config = resolve_rl_training_config(overrides={'self_play': {
            'targeted_start_probability': .1, 'targeted_positions_directory': 'C:/positions'}})
        opts = build_selfplay_overrides(config)
        self.assertEqual(opts['rootDesiredPerChildVisitsCoeff'], 2)
        disabled = replace(config, search=replace(config.search, katago_forced_playouts_coefficient=0))
        self.assertEqual(build_selfplay_overrides(disabled)['rootDesiredPerChildVisitsCoeff'], 0)
        self.assertEqual(opts['hintPosesProb'], .1)
        self.assertEqual(opts['startPosesPolicyInitAreaProp'], 0)
        self.assertEqual(opts['sgfCompensateKomiProb'], 0)
        self.assertEqual(opts['komiMean'], 6.5)
        self.assertNotIn('hintPosesDir', build_selfplay_overrides(config, has_model=False))


if __name__ == '__main__':
    unittest.main()
