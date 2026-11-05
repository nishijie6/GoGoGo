"""One official KataGo network trained on several masked board sizes."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import hashlib
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Optional

from .fileio import atomic_open, atomic_write_json, local_timestamp
from .katago_rl import (PROJECT_ROOT, KataGoRLRunner, KataGoRLRunnerError,
                        build_selfplay_overrides)
from .katago_mixed_health import build_mixed_health
from .katago_promotion import PromotionConfig, PromotionManager
from .replay_accounting import ReplayRowLedger
from .rl_config import RLTrainingConfig, resolve_rl_training_config
from .rl_metrics import parse_model_name, parse_selfplay_output
from .rl_metric_store import RLMetricStore
from .rl_retention import apply_retention_plan, build_retention_plan

DEFAULT_MIXED_PROFILE = PROJECT_ROOT / "config" / "katago_mixed.m4.json"
SUPPORTED_MIXED_BOARDS = (7, 8, 9, 13, 19)


@dataclass(frozen=True)
class MixedExplorationConfig:
    start_positions_dir: Optional[Path] = None
    start_probability: float = 0.0
    early_fork_probability: float = 0.0
    fork_probability: float = 0.0
    policy_init: bool = False


def _load_exploration(raw, config_path: Path) -> Optional[MixedExplorationConfig]:
    if raw is None:
        return None
    keys = {"start_positions_dir", "start_probability", "early_fork_probability", "fork_probability", "policy_init"}
    if not isinstance(raw, dict) or set(raw) - keys:
        raise ValueError("exploration 包含未知字段")
    for name, upper in (("start_probability", 1), ("early_fork_probability", .5), ("fork_probability", .5)):
        value = raw.get(name, 0.0)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"exploration.{name} 必须在 0 到 {upper} 之间")
    policy = raw.get("policy_init", False)
    if type(policy) is not bool:
        raise ValueError("exploration.policy_init 必须是布尔值")
    directory = raw.get("start_positions_dir")
    if directory is not None:
        if not isinstance(directory, str) or any(c in directory for c in "\n\r\0"):
            raise ValueError("开局目录路径无效")
        directory = Path(directory)
        directory = (config_path.parent / directory).resolve() if not directory.is_absolute() else directory.resolve()
        if not directory.is_dir() or not any(p.is_file() and p.suffix in {".sgf", ".sgfs"} for p in directory.rglob("*")):
            raise ValueError("开局目录必须包含 SGF 棋谱")
    if raw.get("start_probability", 0) > 0 and directory is None:
        raise ValueError("启用 SGF 起点必须指定 start_positions_dir")
    return MixedExplorationConfig(directory, raw.get("start_probability", 0.0),
                                  raw.get("early_fork_probability", 0.0), raw.get("fork_probability", 0.0), policy)


@dataclass(frozen=True)
class MixedBoardConfig:
    board_sizes: tuple[int, ...]
    board_weights: tuple[float, ...]
    shuffle_rows_per_file: int
    training: RLTrainingConfig
    exploration: Optional[MixedExplorationConfig] = None
    promotion: Optional[PromotionConfig] = None
    mps_game_threads: Optional[int] = None
    mps_hybrid_inference: bool = False


def load_mixed_config(path: Path) -> MixedBoardConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    keys = {"schema_version", "board_sizes", "board_weights", "shuffle_rows_per_file", "training"}
    if (not isinstance(raw, dict) or not keys <= raw.keys()
            or raw.keys() - keys - {"exploration", "promotion", "mps_game_threads", "mps_hybrid_inference"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
        raise ValueError("混训配置字段或 schema_version 无效")
    boards, weights = raw["board_sizes"], raw["board_weights"]
    if (not isinstance(boards, list) or not boards
            or any(type(b) is not int or b not in SUPPORTED_MIXED_BOARDS for b in boards)
            or len(set(boards)) != len(boards)):
        raise ValueError("混训棋盘必须是不重复的 7、8、9、13、19 路尺寸")
    if (not isinstance(weights, list) or len(weights) != len(boards)
            or any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights)):
        raise ValueError("每种棋盘都必须有一个有限、正数的抽样权重")
    rows = raw["shuffle_rows_per_file"]
    if type(rows) is not int or rows < 1:
        raise ValueError("shuffle_rows_per_file 必须是正整数")
    training = raw["training"]
    if (not isinstance(training, dict) or set(training) != {"schema_version", "preset", "overrides"}
            or type(training["schema_version"]) is not int or training["schema_version"] != 1):
        raise ValueError("training 必须是完整的版本 1 训练配置")
    config = resolve_rl_training_config(preset=training["preset"], overrides=training["overrides"])
    if config.game.board_size < max(boards):
        raise ValueError("训练张量尺寸不能小于最大棋盘（19 路需要 19×19 张量）")
    if rows < config.optimizer.batch_size:
        raise ValueError("每个打乱分片至少需要容纳一个训练批次")
    if config.self_play.joseki_fraction:
        raise ValueError("KataGo 混训不支持普通版 joseki_fraction")
    promotion = PromotionConfig.load(raw["promotion"], boards) if "promotion" in raw else None
    mps_game_threads = raw.get("mps_game_threads")
    if "mps_game_threads" in raw and (type(mps_game_threads) is not int or not 1 <= mps_game_threads <= 32):
        raise ValueError("mps_game_threads 必须是 1 到 32 的整数")
    mps_hybrid_inference = raw.get("mps_hybrid_inference", False)
    if type(mps_hybrid_inference) is not bool:
        raise ValueError("mps_hybrid_inference 必须是布尔值")
    return MixedBoardConfig(tuple(boards), tuple(weights), rows, config,
                            _load_exploration(raw.get("exploration"), path), promotion, mps_game_threads,
                            mps_hybrid_inference)


def write_engine_config(source: Path, destination: Path, overrides: dict) -> Path:
    """Write CSV-valued options as config lines, not comma-delimited CLI overrides."""
    text = source.read_text(encoding="utf-8")
    for key, value in overrides.items():
        value = str(value).lower() if isinstance(value, bool) else str(value)
        line = f"{key} = {value}"
        pattern = rf"(?m)^{re.escape(key)}\s*=.*$"
        text = re.sub(pattern, lambda _: line, text) if re.search(pattern, text) else text + "\n" + line + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    from .fileio import atomic_write_text
    atomic_write_text(destination, text)
    return destination


class MixedBoardRunner(KataGoRLRunner):
    """Reuse the checkpointed single-machine loop with a fixed mixed-data contract."""

    def __init__(self, config_path: Path = DEFAULT_MIXED_PROFILE, *,
                 project_root: Path = PROJECT_ROOT, run_root_override: Optional[Path] = None,
                 hardware_profile=None):
        config_path = config_path if config_path.is_absolute() else project_root / config_path
        self.mixed = load_mixed_config(config_path)
        super().__init__(config_path, project_root, run_root_override, hardware_profile,
                         training_config=self.mixed.training)
        self.metric_store = RLMetricStore(self.run_root, allow_mixed_boards=True,
            standard=(self.mixed.promotion.standard if self.mixed.promotion else
                      getattr(self.config.evaluation, "standard", None)))
        from .katago_bootstrap import BootstrapManager
        self.bootstrap = BootstrapManager(self)
        self.promotion = PromotionManager(self, self.mixed.promotion) if self.mixed.promotion else None

    def manifest_identity(self) -> dict:
        identity = {"board_sizes": list(self.mixed.board_sizes),
                    "board_weights": list(self.mixed.board_weights),
                    "data_board_len": self.config.game.board_size,
                    "model_kind": self.model_kind, "rules": self.config.to_dict()["game"]}
        exploration = self.mixed.exploration
        if exploration is not None:
            hashes = {}
            if exploration.start_positions_dir is not None:
                hashes = {str(p.relative_to(exploration.start_positions_dir)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(exploration.start_positions_dir.rglob("*"))
                          if p.is_file() and p.suffix in {".sgf", ".sgfs"}}
            identity["exploration"] = {"start_probability":exploration.start_probability,
                "start_positions_sha256":hashes, "early_fork_probability":exploration.early_fork_probability,
                "fork_probability":exploration.fork_probability, "policy_init":exploration.policy_init}
        return identity

    def ensure_manifest(self) -> None:
        if self.promotion is None and (self.run_root / "promotion/state.json").exists():
            raise KataGoRLRunnerError("该目录已有晋级状态，续训必须使用包含 promotion 的配置")
        identity = self.manifest_identity()
        path = self.run_root / "mixed_manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")).get("identity") != identity:
                raise ValueError("混训清单与现有目录不一致，不能混用不兼容的数据或模型")
            return
        populated = (any(self.models_dir.glob("*/model.bin*"))
                     or any(self.selfplay_dir.glob("**/tdata/*.npz"))
                     or (self.run_root / "train/gogogo/checkpoint.ckpt").exists()
                     or any(self.shuffled_dir.glob("*/train/*.npz")))
        if populated:
            raise ValueError("已有训练数据但缺少混训清单；请选择新的独立目录")
        self.run_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"schema_version": 1, "created_at": local_timestamp(),
                                "identity": identity, "initial_config": self.config.to_dict()})

    def selfplay_overrides(self, *, visits=None, smoke=False, has_model=True) -> dict:
        overrides = build_selfplay_overrides(self.config, visits=visits, smoke=smoke,
                                             has_model=has_model, hardware_profile=self.hardware_profile)
        cheap_visits = self.config.search.katago_cheap_search_visits
        if smoke or cheap_visits is None:
            cheap_visits = self.config.search.fast_simulations_per_move
        if self.config.hardware.device == "mps":
            overrides = {k:v for k,v in overrides.items() if not k.startswith("cuda")}
            overrides["metalDeviceToUseThread0"] = 0
            if not smoke and self.mixed.mps_game_threads is not None:
                overrides["numGameThreads"] = self.mixed.mps_game_threads
            if self.mixed.mps_hybrid_inference and has_model and not smoke:
                overrides.update(numNNServerThreadsPerModel=4, metalDeviceToUseThread1=0,
                                 metalDeviceToUseThread2=100, metalDeviceToUseThread3=100)
        overrides.update({
            "bSizes": ",".join(map(str, self.mixed.board_sizes)),
            "bSizeRelProbs": ",".join(f"{w:g}" for w in self.mixed.board_weights),
            "dataBoardLen": self.config.game.board_size,
            "maxBoardSizeForNNBuffer": self.config.game.board_size,
            "requireMaxBoardSize": False,
            "cheapSearchProb": 1 - self.config.search.full_search_probability,
            "cheapSearchVisits": min(cheap_visits, overrides["maxVisits"]),
            "estimateLeadProb": 0.0,
        })
        exploration = self.mixed.exploration
        if exploration is not None:
            overrides.update({"startPosesProb":exploration.start_probability,
                "earlyForkGameProb":exploration.early_fork_probability,
                "forkGameProb":exploration.fork_probability, "initGamesWithPolicy":exploration.policy_init})
            if exploration.start_positions_dir is not None:
                overrides["startPosesFromSgfDir"] = str(exploration.start_positions_dir)
                # KataGo requires these two keys whenever a SGF directory is
                # configured; the upstream template leaves them commented out.
                overrides.update(startPosesLoadProb=1.0, startPosesTurnWeightLambda=0.0,
                                 startPosesPolicyInitAreaProp=0.0)
        return overrides

    def selfplay(self, *, games=None, visits=None, smoke=False):
        self.ensure_manifest()
        self._require_runtime()
        self._ensure_directories()
        count = (4 if smoke else self.config.self_play.games_per_iteration) if games is None else games
        if type(count) is not int or count < 1:
            raise ValueError("自我对弈局数必须是正整数")
        champion_models = self.promotion.actor_models() if self.promotion else None
        actor_models = champion_models or self.bootstrap.models_dir()
        if (smoke and self.promotion and champion_models is None and actor_models == self.models_dir
                and any((parse_model_name(path.parent.name) or (0, 0))[0] > 0
                        for path in actor_models.glob("*/model.bin.gz"))):
            raise KataGoRLRunnerError("已有训练导出但尚未登记冠军；请在独立目录运行 smoke")
        has_model = any(actor_models.glob("*/model.bin.gz"))
        targeted_directory = self._targeted_positions_directory(has_model)
        overrides = self.selfplay_overrides(visits=8 if smoke and visits is None else visits, smoke=smoke,
                                            has_model=has_model)
        if targeted_directory is not None:
            overrides["hintPosesDir"] = str(targeted_directory)
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        cfg = write_engine_config(self.selfplay_config, self.run_root / "configs" / f"selfplay-{stamp}.cfg", overrides)
        output = self._run([self.katago_executable, "selfplay", "-max-games-total", count,
                            "-output-dir", self.selfplay_dir, "-models-dir", actor_models,
                            "-config", cfg], log_name=f"selfplay-{stamp}.log")
        if champion_models is not None:
            self.promotion.inspect_live_selfplay()
        return parse_selfplay_output(output)

    def _promotion_probe(self, model: Path, root: Path, board: int, games: int) -> None:
        self._require_runtime()
        output = root / "selfplay"
        output.mkdir(parents=True, exist_ok=True)
        overrides = self.selfplay_overrides(has_model=True)
        overrides.update(bSizes=board, bSizeRelProbs=1)
        cfg = write_engine_config(self.selfplay_config, root / f"probe-{board}.cfg", overrides)
        self._run([self.katago_executable, "selfplay", "-max-games-total", games,
                   "-output-dir", output, "-models-dir", model.parent.parent,
                   "-config", cfg], log_name=str(root / f"probe-{board}.log"))

    def _shuffle_rows_per_file(self, batch_size=None) -> int:
        return self.mixed.shuffle_rows_per_file

    def board_statistics(self) -> dict:
        """Validate newly observed NPZs and cache per-size counts across bounded replay."""
        import numpy as np
        size = self.config.game.board_size
        cache_path = self.run_root / "board_inventory.json"
        old = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        inventory, total = {}, Counter({str(b):0 for b in self.mixed.board_sizes})
        for path in sorted(self.selfplay_dir.glob("**/tdata/*.npz")):
            stat = path.stat()
            key = str(path.relative_to(self.run_root))
            signature = [stat.st_size, stat.st_mtime_ns]
            cached = old.get(key)
            if cached and cached.get("signature") == signature:
                counts = cached["rows_by_board"]
            else:
                with np.load(path, allow_pickle=False) as data:
                    packed = data["binaryInputNCHWPacked"]
                    if (packed.ndim != 3 or packed.shape[2] != (size*size+7)//8
                            or data["valueTargetsNCHW"].shape[-2:] != (size,size)):
                        raise ValueError(f"训练数据张量尺寸不匹配：{path}")
                    areas = np.unpackbits(packed[:,0,:], axis=1)[:,:size*size].sum(axis=1)
                    valid = {b*b:str(b) for b in self.mixed.board_sizes}
                    if not set(areas.tolist()) <= set(valid):
                        raise ValueError(f"训练数据包含未配置的棋盘掩码：{path}")
                    counts = dict(Counter(valid[int(area)] for area in areas))
            inventory[key] = {"signature":signature, "rows_by_board":counts}
            total.update(counts)
        games = Counter({str(b):0 for b in self.mixed.board_sizes})
        for path in self.selfplay_dir.glob("**/*.sgfs"):
            games.update(re.findall(r"SZ\[(\d+)\]", path.read_text(encoding="utf-8")))
        if self.run_root.exists():
            atomic_write_json(cache_path, inventory)
        result = {"updated_at":local_timestamp(), "data_board_len":size,
                  "rows_by_board":dict(total), "games_by_board":dict(games),
                  "note":"开局抽样权重不代表训练行比例；行数只统计当前保留的回放"}
        if self.run_root.exists():
            atomic_write_json(self.run_root / "board_statistics.json", result)
        return result

    def shuffle(self, *, min_rows=None, smoke=False, training_batch_size=None, new_data_rows=None):
        self.ensure_manifest()
        self.board_statistics()
        return super().shuffle(min_rows=min_rows, smoke=smoke,
                               training_batch_size=training_batch_size, new_data_rows=new_data_rows)

    def initialize_model(self, *, publish: bool = True) -> None:
        """Prepare the learner seed, or recover a completed training checkpoint.

        Fresh self-play must use KataGo's native random bootstrap. Keep its
        learner seed outside the export queue until learning produces a model.
        """
        self.ensure_manifest()
        self._require_runtime()
        self._ensure_directories()
        if any(self.models_dir.glob("*/model.bin.gz")):
            return
        import torch
        checkpoint = self.run_root / "train/gogogo/checkpoint.ckpt"
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            counter = state["train_state"]
            name = f"gogogo-s{counter['global_step_samples']}-d{counter['total_num_data_rows']}"
            target = self.run_root / "torchmodels_toexport" / name / "model.ckpt"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(checkpoint, target)
        else:
            if not publish and (self.run_root / "torchmodels_toexport/gogogo-s0-d0/model.ckpt").exists():
                raise KataGoRLRunnerError(
                    "发现旧版未发布的种子检查点；请使用新的独立目录进行原生随机自我对弈预热"
                )
            target = self.run_root / ("torchmodels_toexport/gogogo-s0-d0/model.ckpt"
                                      if publish else "initial_model/model.ckpt")
            if not target.exists():
                previous_path = list(sys.path)
                try:
                    sys.path.insert(0, str(self.python_source))
                    from copy import deepcopy
                    from katago.train.model_pytorch import Model
                    from katago.train import modelconfigs
                    from torch.optim.swa_utils import AveragedModel
                    torch.manual_seed(self.config.runtime.random_seed)
                    model_config = deepcopy(modelconfigs.config_of_name[self.model_kind])
                    model = Model(model_config, self.config.game.board_size)
                    model.initialize()
                    swa = AveragedModel(model)
                    swa.n_averaged.fill_(1)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with atomic_open(target, "wb") as output:
                        torch.save({"model":model.state_dict(), "swa_model":swa.state_dict(),
                                    "config":model_config}, output)
                finally:
                    sys.path[:] = previous_path
        if publish or checkpoint.exists():
            self.export_new_models()

    def cycle(self, **kwargs) -> None:
        self.ensure_manifest()
        bootstrap_state = self.bootstrap.read_state()
        checkpoint = self.run_root / "train/gogogo/checkpoint.ckpt"
        if not checkpoint.exists():
            candidates = [*self.models_dir.glob("*/model.bin.gz"),
                          *(self.run_root / "torchmodels_toexport").glob("*/model.ckpt")]
            if any((match := re.search(r"-s([0-9]+)-d[0-9]+(?:\.tmp(?:-[0-9]+)?)?$", path.parent.name))
                   and int(match.group(1)) > 0 for path in candidates):
                raise KataGoRLRunnerError(
                    "已有训练模型但缺少完整训练检查点；请恢复 train/gogogo/checkpoint.ckpt，"
                    "或使用新的独立目录，不能从初始种子重置学习器"
                )
        bootstrap = not checkpoint.exists() and not any(self.models_dir.glob("*/model.bin.gz"))
        self.initialize_model(publish=not bootstrap)
        if bootstrap and bootstrap_state is None:
            self.bootstrap.start()
        if not checkpoint.exists() and kwargs.get("initial_checkpoint") is None:
            kwargs["initial_checkpoint"] = self.run_root / (
                "initial_model/model.ckpt" if bootstrap
                else "torchmodels_toexport/gogogo-s0-d0/model.ckpt")
        self.run_root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.run_root).free < 10 * 1024**3:
            raise KataGoRLRunnerError("磁盘可用空间不足 10 GiB，混训已停止并保留检查点")
        promotion = self.promotion if not kwargs.get("smoke") else None
        if promotion:
            promotion.advance()
        super().cycle(**kwargs)
        # Use the existing write-ahead replay ledger so pruning preserves data watermarks.
        ledger = ReplayRowLedger(self.run_root)
        ledger.sync()
        plan = build_retention_plan(self.run_root, protected_models=(),
                                    keep_models=self.config.runtime.keep_checkpoints,
                                    keep_shuffles=2,
                                    replay_window_rows=self.config.optimizer.replay_buffer_capacity)
        apply_retention_plan(plan, self.run_root)
        ledger.sync()
        self.bootstrap.after_cycle()
        if promotion:
            promotion.advance()
        print(json.dumps({"mixed_boards":self.board_statistics()}, ensure_ascii=False), flush=True)
        standard = self.mixed.promotion.standard if self.mixed.promotion else None
        health = build_mixed_health(self.run_root, self.mixed.board_sizes,
                                   **({"standard": standard} if standard else {}))
        print(json.dumps({"mixed_health": {"status": health["status"],
                         "report": str(self.run_root / "health_by_board.json"),
                         "by_board": {b: row["status"] for b, row in health["by_board"].items()}}},
                         ensure_ascii=False), flush=True)

    def status(self):
        result = super().status()
        result["bootstrap"] = self.bootstrap.read_state()
        if self.promotion:
            result["promotion"] = self.promotion.read_state()
        result["mixed_boards"] = self.board_statistics()
        health_path = self.run_root / "health_by_board.json"
        # Status reports the last completed diagnostic snapshot. mixed-health
        # explicitly refreshes it; cycles scan SGFs once, after all stages.
        result["mixed_health"] = json.loads(health_path.read_text(encoding="utf-8")) if health_path.exists() else None
        health = result["mixed_health"]
        health_summary = ({"status": health["status"], "updated_at": health["updated_at"],
                          "report": str(health_path),
                          "by_board": {b: row["status"] for b, row in health["by_board"].items()}}
                         if health is not None else None)
        print(json.dumps({"mixed_boards":result["mixed_boards"], "mixed_health": health_summary,
                          "bootstrap": result["bootstrap"], "promotion": result.get("promotion")},
                         ensure_ascii=False), flush=True)
        return result
