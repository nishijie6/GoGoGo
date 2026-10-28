"""Native MCTS ablation on frozen local weights and independent teacher labels.

The available native checkpoint is a one-iteration smoke model. This measures
search mechanics and target quality, not the strength of the live KataGo run.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from weiqi.fileio import atomic_write_json, sha256_file
from weiqi.rl.network import PolicyValueNet
from weiqi.rl.search import search
from weiqi.rl.state import Position
from weiqi.rl.storage import load_checkpoint
from weiqi.rl.eval_positions import read_suite, read_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'training_runs/targeted-forced-20261028/native')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'training_runs/merge-validation-20260929/native/latest.pt'
    payload, config = load_checkpoint(source)
    torch.set_num_threads(2)
    torch.manual_seed(20261028)
    net = PolicyValueNet(config)
    net.load_state_dict(payload['model'])
    net.eval()
    suite_path = ROOT / 'config/rl_eval_positions_9x9.json'
    teacher_path = ROOT / 'config/rl_eval_teacher_9x9.json'
    suite = read_suite(suite_path)
    labels = read_teacher(teacher_path, suite)
    positions = []
    for row in suite['positions']:
        state = Position.new(9, 6.5)
        for color, action in row['moves']:
            if state.game.current_player != color:
                raise RuntimeError('Bad frozen position history')
            state = state.play(action)
        target = np.maximum(np.asarray(labels['positions'][row['id']]['policy'], dtype=np.float32), 0)
        target *= state.legal
        target /= target.sum()
        positions.append((row, state, target))

    def evaluate(features):
        with torch.inference_mode():
            p, v = net(torch.from_numpy(features[None]))
        return p[0].numpy(), float(v[0])

    def batch(features):
        with torch.inference_mode():
            p, v = net(torch.from_numpy(features))
        return p.numpy(), v.numpy()

    results = {}
    arms = [('baseline', 0., False), ('forced-only', 2., False), ('forced-pruned', 2., True)]
    for name, coefficient, prune in arms:
        search_config = replace(config.search, leaf_batch_size=4,
            forced_playouts_coefficient=coefficient, policy_target_pruning=prune)
        started = time.perf_counter()
        records, examples = [], []
        # Same positions, budgets and independent root-noise seeds in each arm.
        for replica in range(2):
            for row, state, teacher in positions:
                target = np.empty(82, np.float32)
                info = {}
                policy, _ = search(state, evaluate, search_config,
                    np.random.default_rng(20261028 + replica * 1000 + int(row['id'][1:])),
                    simulations=64, add_noise=True, evaluate_batch=batch,
                    target_policy_out=target, diagnostics=info)
                records.append({'id': row['id'], 'source_game': row['source_game'], 'replica': replica,
                    'teacher_target_ce': float(-(teacher * np.log(np.maximum(target, 1e-8))).sum()),
                    'teacher_probability_of_played_move': float(teacher[int(policy.argmax())]),
                    **info})
                examples.append((row['source_game'], state.features(), target, teacher))
        seconds = time.perf_counter() - started
        # Equal short supervised policy updates on generated targets; whole
        # source games are held out. No rollout values are invented.
        training = [row for row in examples if row[0] % 3 != 0]
        heldout = [row for row in examples if row[0] % 3 == 0]
        probes = []
        for replica in (1, 2):
            torch.manual_seed(20261028 + replica)
            probe = copy.deepcopy(net)
            probe.train()
            optimizer = torch.optim.AdamW(probe.parameters(), lr=config.optimizer.learning_rate,
                                           weight_decay=config.optimizer.weight_decay)
            if payload.get('optimizer'):
                optimizer.load_state_dict(copy.deepcopy(payload['optimizer']))
            features = torch.from_numpy(np.stack([row[1] for row in training]))
            targets = torch.from_numpy(np.stack([row[2] for row in training]))
            generator = np.random.default_rng(20261028 + replica)
            for _ in range(32):
                indices = generator.integers(0, len(training), 32)
                logits, _ = probe(features[indices])
                loss = -(targets[indices] * F.log_softmax(logits, dim=1)).sum(1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), config.optimizer.gradient_clip_norm)
                optimizer.step()
            probe.eval()
            with torch.inference_mode():
                logits, _ = probe(torch.from_numpy(np.stack([row[1] for row in heldout])))
                teachers = torch.from_numpy(np.stack([row[3] for row in heldout]))
                ce = -(teachers * F.log_softmax(logits, dim=1)).sum(1)
            probes.append({'replica': replica, 'heldout_teacher_policy_ce': float(ce.mean()),
                'heldout_source_games': sorted({row[0] for row in heldout}),
                'training_source_games': sorted({row[0] for row in training})})
        results[name] = {'search_seconds': seconds, 'searches': len(records), 'simulations': 64,
            'forced_playouts': sum(r['forced_playouts'] for r in records),
            'pruned_visits': sum(r['pruned_visits'] for r in records),
            'visited_actions_mean': float(np.mean([r['visited_actions'] for r in records])),
            'teacher_target_ce': float(np.mean([r['teacher_target_ce'] for r in records])),
            'teacher_probability_of_played_move': float(np.mean([r['teacher_probability_of_played_move'] for r in records])),
            'policy_training_probes': probes, 'records': records}
        atomic_write_json(args.output / 'results.json', {'source_checkpoint': str(source),
            'source_sha256': sha256_file(source), 'native_training_iteration': payload['iteration'],
            'suite_sha256': sha256_file(suite_path), 'teacher_sha256': sha256_file(teacher_path),
            'limits': 'One-iteration native smoke checkpoint; short search/policy probes, not Elo.',
            'arms': results})
        print('NATIVE_RESULT', name, json.dumps({k:v for k,v in results[name].items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
