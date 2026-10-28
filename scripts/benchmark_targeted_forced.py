"""Isolated targeted-start and forced-playout experiments on frozen models."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from weiqi.fileio import atomic_write_json, local_timestamp, sha256_file
from weiqi.katago_hardware import detect_hardware
from weiqi.katago_rl import KataGoRLRunner, build_selfplay_overrides, format_katago_overrides
from weiqi.replay_accounting import ReplayRowLedger
from weiqi.rl_curriculum import GlobalCurriculumLock
from weiqi.rl_curriculum_runtime import CurriculumRuntime
from weiqi.rl_metrics import checkpoint_metrics, parse_selfplay_output, parse_sgfs
from weiqi.rl_targeted_positions import mine_training_files
from scripts.benchmark_rl_efficiency import GpuMonitor, optimizer_steps, validate_checkpoint_update
from scripts.benchmark_schedule_replay import ArrayWriter, read_arrays, game_id, held_out_game, MemoryMonitor

CONFIG = ROOT / 'config/rl_training.rtx5070ti.curriculum.19x19.json'
ARMS = ('baseline', 'targeted10', 'no-forced')


def save(output, name, data):
    atomic_write_json(output / name, data)


def freeze(output, previous_samples):
    """Stop the verified curriculum only after its current checkpoint is saved."""
    import psutil
    if (output / 'frozen.json').exists():
        raise RuntimeError('Refusing to replace a frozen experiment; use a new output directory')
    if not (ROOT / 'training_runs/curriculum/AUTO_RECOVERY_PAUSED').is_file():
        raise RuntimeError('Pause automatic recovery before freezing the production trainer')
    state_path = ROOT / 'training_runs/curriculum/state.json'
    while True:
        state = json.loads(state_path.read_text(encoding='utf8'))
        stage = state['stages']['19x19']
        owners = []
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            args = p.info['cmdline'] or []
            if (p.info['name'] == 'python.exe' and 'curriculum' in args
                    and any(Path(a).name == 'train_rl.py' for a in args)):
                owners.append(p)
        children = [child for p in owners for child in p.children(recursive=True)]
        safe = state['phase'] == 'evaluating' or (stage['samples'] > previous_samples and
            any(child.name() == 'katago.exe' and 'selfplay' in child.cmdline() for child in children))
        if safe:
            records = [{'pid': p.pid, 'command': p.cmdline()} for p in owners]
            save(output, 'before/stopped-processes.json', records)
            for p in sorted(owners, key=lambda p: len(p.parents()), reverse=True):
                try:
                    p.terminate()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(owners, timeout=10)
            break
        print('WAIT_BOUNDARY', stage['samples'], state['phase'], flush=True)
        time.sleep(15)
    runtime = CurriculumRuntime(ROOT / 'config/rl_curriculum.rtx5070ti.json')
    with GlobalCurriculumLock(runtime.state_root):
        state = runtime.store.load()
        source = runtime._stage_runner(state)
        with source.lock():
            checkpoint = source.run_root / 'train/gogogo/checkpoint.ckpt'
            before = output / 'before'
            shutil.copy2(checkpoint, before / 'checkpoint.ckpt')
            shutil.copy2(state_path, before / 'state.json')
            shutil.copy2(source.run_root / 'replay_rows.json', before / 'replay_rows.json')
            shutil.copy2(CONFIG, before / 'stage-config.json')
            model = runtime._model_path(source, state.active.latest_model)
            shutil.copy2(model, before / 'model.bin.gz')
            for relative in json.loads((before / 'replay_rows.json').read_text(encoding='utf8'))['files']:
                target = before / 'replay' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source.run_root / relative, target)
            metrics = checkpoint_metrics(checkpoint)
            if metrics['trained_samples'] != state.active.samples:
                raise RuntimeError('Saved checkpoint and curriculum disagree')
            save(output, 'frozen.json', {'timestamp': local_timestamp(),
                'model': state.active.latest_model, 'samples': state.active.samples,
                'checkpoint_sha256': sha256_file(checkpoint), 'model_sha256': sha256_file(model),
                'batch': state.active.autotune_batch, 'metrics': metrics})
            print('FROZEN', json.dumps(metrics), flush=True)


def mine(output):
    files = sorted((ROOT / 'training_runs/19x19/selfplay').glob('*/sgfs/*.sgfs'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    training, validation, rejected = mine_training_files(files)
    if len(training) < 128 or len(validation) < 32:
        raise RuntimeError('Insufficient independent weak positions')
    directory = output / 'positions'
    directory.mkdir(exist_ok=False)
    for name, positions in [('training.startposes.txt', training), ('validation.jsonl', validation)]:
        (directory / name).write_text(''.join(json.dumps(p) + '\n' for p in positions), encoding='utf8')
    assert not ({p['game_sha256'] for p in training} & {p['game_sha256'] for p in validation})
    save(output, 'mining.json', {'training_games': len(training), 'validation_games': len(validation),
        'rejected': rejected, 'rule': 'max consecutive annotated black-win-probability change per game',
        'minimum_swing': min(p['swing'] for p in training), 'maximum_swing': max(p['swing'] for p in training),
        'note': 'Mining positions come only from self-play, not official evaluation matches.'})


def selfplay(output):
    frozen = json.loads((output / 'frozen.json').read_text(encoding='utf8'))
    profile = detect_hardware()
    results = {}
    for arm in ARMS:
        runner = KataGoRLRunner(CONFIG, run_root_override=output / arm / 'generation', hardware_profile=profile)
        target = runner.models_dir / frozen['model'] / 'model.bin.gz'
        target.parent.mkdir(parents=True, exist_ok=False)
        shutil.copy2(output / 'before/model.bin.gz', target)
        runner._ensure_directories()
        opts = build_selfplay_overrides(runner.config, hardware_profile=profile)
        opts.update(rootDesiredPerChildVisitsCoeff=0 if arm == 'no-forced' else 2,
                    switchNetsMidGame=False)
        if arm == 'targeted10':
            opts.update(hintPosesProb=.1, hintPosesDir=str(output / 'positions'),
                        startPosesPolicyInitAreaProp=0.0)
        started = time.perf_counter()
        with GpuMonitor(profile.gpu_index) as gpu, MemoryMonitor() as memory:
            log = runner._run([runner.katago_executable, 'selfplay', '-max-games-total', 512,
                '-output-dir', runner.selfplay_dir, '-models-dir', runner.models_dir,
                '-config', runner.selfplay_config, '-override-config', format_katago_overrides(opts)],
                log_name='comparison-selfplay.log')
        sgfs = parse_sgfs(list(runner.selfplay_dir.glob('*/sgfs/*.sgfs')))
        result = {'seconds': time.perf_counter() - started, 'engine': parse_selfplay_output(log),
                  'sgf': sgfs, 'gpu': gpu.metrics(), 'memory': memory.metrics(), 'overrides': opts}
        results[arm] = result
        save(output, 'selfplay-results.json', results)
        print('SELFPLAY_RESULT', arm, json.dumps(result), flush=True)


def split_fresh(output):
    if (output / 'fresh-manifests.json').exists():
        raise RuntimeError('Fresh partitions already exist; refusing to overwrite them')
    manifests = {}
    all_training, all_validation = set(), set()
    frozen = json.loads((output / 'frozen.json').read_text(encoding='utf8'))
    for arm in ARMS:
        models = list((output / arm / 'generation/models').glob('*/model.bin.gz'))
        if len(models) != 1 or sha256_file(models[0]) != frozen['model_sha256']:
            raise RuntimeError('Generation arms used different model weights')
        paths = sorted((output / arm / 'generation/selfplay').glob('*/sgfs/*.sgfs'))
        health = parse_sgfs(paths)
        if health['sgf_games'] != 512 or health['damaged_games'] or health['no_result_games']:
            raise RuntimeError(f'{arm}: incomplete or unhealthy game generation: {health}')
        types = {}
        for path in paths:
            for line in path.read_text(encoding='utf8').splitlines():
                header = line.split(';B[', 1)[0].split(';W[', 1)[0]
                if 'KM[6.5]' not in header or 'koPOSITIONALscoreAREAtaxNONEsui0' not in header:
                    raise RuntimeError('Generated game escaped the fixed rule/komi domain')
                kind = re.search(r'gtype=([a-z]+)', header)
                key = kind[1] if kind else 'unknown'
                types[key] = types.get(key, 0) + 1
        if arm == 'targeted10' and types.get('sgfpos', 0) < 10:
            raise RuntimeError('Targeted starts were not actually sampled')
        train = ArrayWriter(output / arm / 'fresh-train')
        val = ArrayWriter(output / arm / 'fresh-holdout')
        groups = {'train': set(), 'validation': set()}
        for path in sorted((output / arm / 'generation/selfplay').glob('*/tdata/*.npz')):
            arrays = read_arrays(path)
            indices = {'train': [], 'validation': []}
            for i, row in enumerate(arrays['globalTargetsNC']):
                key = 'validation' if held_out_game(row, 26) else 'train'
                indices[key].append(i)
                groups[key].add(game_id(row))
            train.add(arrays, indices['train'])
            val.add(arrays, indices['validation'])
        train.flush()
        val.flush()
        if groups['train'] & groups['validation'] or val.rows < 2560:
            raise RuntimeError('Bad whole-game validation partition')
        all_training.update(groups['train'])
        all_validation.update(groups['validation'])
        manifests[arm] = {'train_rows': train.rows, 'validation_rows': val.rows,
            'train_games': len(groups['train']), 'validation_games': len(groups['validation']),
            'generated_game_types': types, 'sgf_health': health}
    if all_training & all_validation:
        raise RuntimeError('Cross-arm game leakage')
    # All three evaluation cohorts remain withheld from every arm.
    save(output, 'fresh-manifests.json', manifests)


def train(output):
    import torch
    torch.set_num_threads(2)
    if any((output / arm / 'train-r1').exists() for arm in ARMS):
        raise RuntimeError('Training probes already exist; use a new experiment directory')
    profile = detect_hardware()
    frozen = json.loads((output / 'frozen.json').read_text(encoding='utf8'))
    manifests = json.loads((output / 'fresh-manifests.json').read_text(encoding='utf8'))
    checkpoint = output / 'before/checkpoint.ckpt'
    old_steps = optimizer_steps(checkpoint)
    # Equal optimizer budget at production updates/game, bounded by 4x fresh.
    batch = frozen['batch']
    steps = min(64, min(4 * m['train_rows'] // batch for m in manifests.values()))
    steps = steps // 16 * 16
    if steps < 32:
        raise RuntimeError('Not enough fresh rows for matched training')
    results = []
    for arm in ARMS:
        first_shuffle = None
        for replica in (1, 2):
            runner = KataGoRLRunner(CONFIG, run_root_override=output / arm / f'train-r{replica}',
                                    hardware_profile=profile)
            runner.config = replace(runner.config, optimizer=replace(runner.config.optimizer,
                replay_buffer_capacity=500000, training_steps_per_iteration=steps))
            runner._ensure_directories()
            target = runner.run_root / 'train/gogogo/checkpoint.ckpt'
            shutil.copy2(checkpoint, target)
            started = time.perf_counter()
            if replica == 1:
                ledger = json.loads((output / 'before/replay_rows.json').read_text(encoding='utf8'))
                for relative in ledger['files']:
                    dest = runner.run_root / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    os.link(output / 'before/replay' / relative, dest)
                shutil.copy2(output / 'before/replay_rows.json', runner.run_root / 'replay_rows.json')
                for path in (output / arm / 'fresh-train').glob('*.npz'):
                    dest = runner.selfplay_dir / 'fresh/tdata' / path.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
                    os.utime(dest, None)
                new = ReplayRowLedger(runner.run_root).sync().logical_rows - frozen['metrics']['data_rows']
                assert new == manifests[arm]['train_rows']
                first_shuffle = runner.shuffle(training_batch_size=batch, new_data_rows=new)
            else:
                shutil.copytree(first_shuffle, runner.shuffled_dir / 'frozen', copy_function=os.link)
            shuffle_seconds = time.perf_counter() - started
            started = time.perf_counter()
            runner.train(batch_size=batch, new_data_rows=manifests[arm]['train_rows'])
            elapsed = time.perf_counter() - started
            after = checkpoint_metrics(target)
            advanced = optimizer_steps(target)
            assert after['trained_samples'] - frozen['samples'] == steps * batch
            assert all(advanced[key] - count == steps for key, count in old_steps.items())
            health = validate_checkpoint_update(checkpoint, target)
            exports = runner.export_new_models()
            result = {'arm': arm, 'replica': replica, 'batch': batch, 'steps': steps,
                'samples': steps * batch, 'train_seconds': elapsed, 'shuffle_seconds': shuffle_seconds,
                'exported': [p.name for p in exports], **health}
            results.append(result)
            save(output, 'training-results.json', results)
            print('TRAIN_RESULT', json.dumps(result), flush=True)


def evaluate(output):
    import torch
    from torch.nn import functional as F
    sys.path.insert(0, str(ROOT / 'katago/source/python'))
    from katago.train import load_model, data_processing_pytorch
    torch.set_num_threads(4)
    checkpoints = {'start': output / 'before/checkpoint.ckpt'}
    checkpoints.update({f'{arm}-r{replica}': output / arm / f'train-r{replica}/train/gogogo/checkpoint.ckpt'
                        for arm in ARMS for replica in (1, 2)})
    results = {}
    for name, checkpoint in checkpoints.items():
        raw, model, _ = load_model.load_model(str(checkpoint), use_swa=True, device='cuda', pos_len=19)
        model.eval()
        cohorts = {}
        for cohort in ARMS:
            games = {}
            rows = 0
            for batch in data_processing_pytorch.read_npz_training_data(
                [str(p) for p in sorted((output / cohort / 'fresh-holdout').glob('*.npz'))],
                batch_size=256, world_size=1, rank=0, pos_len=19, device='cuda',
                randomize_symmetries=False, include_meta=model.get_has_metadata_encoder(), model_config=model.config):
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model.postprocess_output(model(batch['binaryInputNCHW'], batch['globalInputNC']))[0]
                target = batch['globalTargetsNC']
                policy = batch['policyTargetsNCMove'][:, 0, :].float()
                policy /= policy.sum(1, keepdim=True).clamp_min(1e-20)
                pw = target[:, 25] * target[:, 26]
                vw = target[:, 25] * (1 - target[:, 35])
                pc = -(policy * F.log_softmax(pred[0][:, 0, :].float(), dim=1)).sum(1)
                vc = -(target[:, :3] * F.log_softmax(pred[1].float(), dim=1)).sum(1)
                amounts = torch.stack((pc * pw, pw, vc * vw, vw), dim=1).cpu().numpy()
                for row, values in zip(target.cpu().numpy(), amounts):
                    key = game_id(row).hex()
                    games[key] = (np.asarray(games.get(key, [0., 0., 0., 0.])) + values).tolist()
                rows += len(target)
            totals = np.asarray(list(games.values())).sum(0)
            cohorts[cohort] = {'rows': rows, 'games': len(games), 'policy_ce': totals[0] / totals[1],
                'value_ce': totals[2] / totals[3], 'per_game': games}
        results[name] = cohorts
        save(output, 'holdout-results.json', results)
        print('HOLDOUT_RESULT', name, json.dumps({k: {a: b for a, b in v.items() if a != 'per_game'}
                                                           for k, v in cohorts.items()}), flush=True)
        del raw, model
        torch.cuda.empty_cache()


def analyze(output):
    data = json.loads((output / 'holdout-results.json').read_text(encoding='utf8'))
    result = {}
    rng = np.random.default_rng(20261028)
    for arm in ('targeted10', 'no-forced'):
        result[arm] = {}
        for cohort in ARMS:
            keys = sorted(data['baseline-r1'][cohort]['per_game'])
            base = np.mean([np.asarray([data[f'baseline-r{r}'][cohort]['per_game'][k] for k in keys])
                            for r in (1, 2)], axis=0)
            other = np.mean([np.asarray([data[f'{arm}-r{r}'][cohort]['per_game'][k] for k in keys])
                             for r in (1, 2)], axis=0)
            metrics = {}
            for offset, name in ((0, 'policy_ce'), (2, 'value_ce')):
                def change(indices):
                    b = base[indices].sum(0)
                    a = other[indices].sum(0)
                    return 100 * ((a[offset] / a[offset+1]) / (b[offset] / b[offset+1]) - 1)
                samples = [change(rng.integers(0, len(keys), len(keys))) for _ in range(2000)]
                metrics[name] = {'change_percent': change(np.arange(len(keys))),
                    'paired_game_bootstrap_95': np.percentile(samples, [2.5, 97.5]).tolist()}
            result[arm][cohort] = metrics
    save(output, 'comparison.json', {'cohorts': result,
        'limits': 'Short proxy comparison conditional on two replicas; not long-horizon Elo.'})
    print('COMPARISON', json.dumps(result), flush=True)


def weak_validation(output):
    """Independent mature-teacher labels on weak games excluded from mining."""
    from weiqi.engine import GoGame
    from weiqi.katago import KataGoEngine
    from weiqi.katago_settings import KataGoSettings
    import weiqi.katago as engine_module
    # Compare raw policy heads in one fixed orientation rather than drawing a
    # new random symmetry for each short-update checkpoint.
    analysis_config = output / 'weak-analysis.cfg'
    text = engine_module.DEFAULT_ANALYSIS_CONFIG.read_text(encoding='utf8')
    text = text.replace('nnRandomize = true', 'nnRandomize = false')
    text = text.replace('numSearchThreadsPerAnalysisThread = 4', 'numSearchThreadsPerAnalysisThread = 1')
    analysis_config.write_text(text, encoding='utf8')
    engine_module.DEFAULT_ANALYSIS_CONFIG = analysis_config
    letters = 'ABCDEFGHJKLMNOPQRST'
    positions = [json.loads(line) for line in
        (output / 'positions/validation.jsonl').read_text(encoding='utf8').splitlines()]
    games = []
    for position in positions:
        game = GoGame(19, komi=6.5, record_undo=False)
        for color, location in zip(position['movePlas'], position['moveLocs']):
            if game.current_player != (1 if color == 'B' else 2):
                raise RuntimeError('Mined player order mismatch')
            if location == 'pass':
                game.pass_turn()
            else:
                column = letters.index(location[0])
                row = 19 - int(location[1:])
                if not game.play(row, column).legal:
                    raise RuntimeError('Mined illegal history')
        games.append(game)
    discovered = KataGoSettings.load()
    teacher_settings = KataGoSettings(executable=str(discovered.resolved_executable()),
                                     model=str(discovered.resolved_model()))
    teacher = KataGoEngine(teacher_settings)
    labels = []
    try:
        for i, game in enumerate(games):
            response = teacher.analyze_position(game, max_visits=800, pv_length=1,
                include_policy=True, include_ownership=False, preserve_history=True)
            visits = np.zeros(362, dtype=np.float64)
            for move in response['moveInfos']:
                loc = move['move']
                index = 361 if loc == 'pass' else (19 - int(loc[1:])) * 19 + letters.index(loc[0])
                visits[index] = move['visits']
            visits /= visits.sum()
            labels.append(visits.tolist())
            if (i+1) % 16 == 0:
                print('WEAK_TEACHER', i+1, len(games), flush=True)
    finally:
        teacher.close()
    save(output, 'weak-teacher.json', {'model': str(teacher_settings.resolved_model()),
        'model_sha256': sha256_file(teacher_settings.resolved_model()), 'visits': 800,
        'analysis_config_sha256': sha256_file(analysis_config), 'nn_randomize': False,
        'search_threads': 1,
        'games': [p['game_sha256'] for p in positions], 'policy_targets': labels,
        'note': 'These weak histories may have been seen by the starting checkpoint, but are excluded from targeted sampling.'})
    models = {'start': output / 'before/model.bin.gz'}
    for result in json.loads((output / 'training-results.json').read_text(encoding='utf8')):
        models[f"{result['arm']}-r{result['replica']}"] = (output / result['arm'] /
            f"train-r{result['replica']}/models" / result['exported'][-1] / 'model.bin.gz')
    results = {}
    for name, path in models.items():
        engine = KataGoEngine(KataGoSettings(executable=str(ROOT / 'katago/katago.exe'), model=str(path)))
        ces, hits, probabilities = [], [], []
        try:
            for game, label in zip(games, labels):
                response = engine.analyze_position(game, max_visits=1, pv_length=1,
                    include_policy=True, include_ownership=False, preserve_history=True)
                policy = np.maximum(np.asarray(response['policy'], dtype=float), 0)
                policy /= policy.sum()
                label = np.asarray(label)
                ces.append(float(-(label * np.log(np.maximum(policy, 1e-12))).sum()))
                hits.append(int(policy.argmax() == label.argmax()))
                probabilities.append(float(label[policy.argmax()]))
        finally:
            engine.close()
        results[name] = {'teacher_policy_ce': float(np.mean(ces)), 'top1_agreement': float(np.mean(hits)),
            'teacher_visit_mass_of_chosen_move': float(np.mean(probabilities)), 'per_game_ce': ces}
        save(output, 'weak-validation.json', results)
        print('WEAK_RESULT', name, json.dumps({k:v for k,v in results[name].items() if k != 'per_game_ce'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'mine', 'selfplay', 'split', 'train', 'evaluate', 'analyze', 'weak'))
    parser.add_argument('--output', type=Path, default=ROOT / 'training_runs/targeted-forced-20261028')
    parser.add_argument('--previous-samples', type=int, default=11486720)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == 'freeze':
        freeze(output, args.previous_samples)
    elif args.mode in ('selfplay', 'train', 'evaluate', 'weak'):
        runtime = CurriculumRuntime(ROOT / 'config/rl_curriculum.rtx5070ti.json')
        with GlobalCurriculumLock(runtime.state_root):
            state = runtime.store.load()
            source = runtime._stage_runner(state)
            with source.lock():
                frozen = json.loads((output / 'frozen.json').read_text(encoding='utf8'))
                source_checkpoint = source.run_root / 'train/gogogo/checkpoint.ckpt'
                if sha256_file(source_checkpoint) != frozen['checkpoint_sha256']:
                    raise RuntimeError('Production checkpoint changed')
                globals()[{'weak': 'weak_validation'}.get(args.mode, args.mode)](output)
                if sha256_file(source_checkpoint) != frozen['checkpoint_sha256']:
                    raise RuntimeError('Production checkpoint changed during experiment')
    else:
        globals()[{'split': 'split_fresh'}.get(args.mode, args.mode)](output)


if __name__ == '__main__':
    main()
