"""Read-only, CPU validation of the native trainer's complete checkpoints.

The archive layer imports this adapter only for native runs, so KataGo handoff
and archive verification still work without the optional PyTorch dependencies.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from .rl_config import RLTrainingConfig, load_rl_training_config, resolve_rl_training_config


_REQUIRED_STATES = {
    "model", "best_model", "optimizer", "scaler", "replay", "numpy_rng",
    "torch_rng", "cuda_rng", "iteration", "training_steps", "best_iteration",
}


def _error(message: str):
    # The archive engine imports this adapter lazily; keep this import lazy too.
    from .rl_handoff import HandoffError
    return HandoffError(message)


def _config_document(config: RLTrainingConfig) -> dict:
    values = config.to_dict()
    return {"schema_version": values.pop("schema_version"),
            "preset": values.pop("preset"), "overrides": values}


def _algorithm_config(config: RLTrainingConfig) -> dict:
    """Strip only explicitly permitted machine-specific differences."""
    values = config.to_dict()
    values.pop("hardware")
    values["runtime"].pop("output_directory")
    for name in ("workers", "inference_batch_size"):
        values["self_play"].pop(name)
    values["optimizer"].pop("batch_size")
    return values


def apply_native_hardware(config: RLTrainingConfig, path: Path | None) -> RLTrainingConfig:
    """Apply the shared hardware allowlist without rewriting a checkpoint."""
    from .rl_handoff import _hardware_overrides
    overrides = _config_document(config)["overrides"]
    for section, changes in _hardware_overrides(path).items():
        overrides[section].update(changes)
    return resolve_rl_training_config(config.preset, overrides)


def _load_and_validate(path: Path, expected_config: Path | None = None):
    try:
        import numpy as np
        import torch
        from .rl.network import PolicyValueNet
        from .rl.storage import ReplayBuffer, fingerprint, load_checkpoint
    except ImportError as error:
        raise _error("普通版接力导出和检查点验收需要 PyTorch/NumPy，请使用已安装训练依赖的环境") from error

    if not path.is_file() or path.is_symlink():
        raise _error(f"普通版续训需要完整 latest.pt 检查点文件：{path}")
    try:
        payload, config = load_checkpoint(path)
        if payload.get("kind") != "training":
            raise ValueError("需要完整 latest.pt 训练检查点，不能使用 candidate.pt/best.pt 模型导出")
        missing = sorted(_REQUIRED_STATES - payload.keys())
        if missing:
            raise ValueError(f"完整训练检查点缺少状态：{', '.join(missing)}")
        for name in ("iteration", "training_steps", "best_iteration", "untrained_samples"):
            value = payload.get(name, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f"检查点 {name} 必须是非负整数")
        if payload["best_iteration"] > payload["iteration"]:
            raise ValueError("检查点 best_iteration 不能超过 iteration")
        for name in ("model", "best_model", "optimizer", "scaler", "replay", "numpy_rng"):
            if not isinstance(payload[name], dict):
                raise ValueError(f"检查点 {name} 必须是状态对象")
        if expected_config is not None:
            imported = load_rl_training_config(expected_config)
            if _algorithm_config(config) != _algorithm_config(imported):
                raise ValueError("导入配置与检查点训练算法不一致；只允许输出目录和硬件参数改变")

        # Model construction consumes random numbers. Preserve caller CPU RNG
        # and explicitly avoid enumerating/initializing any CUDA devices.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            model = PolicyValueNet(config)
            model.load_state_dict(payload["model"], strict=True)
            for state_name in ("model", "best_model"):
                state = payload[state_name]
                if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all()
                           for value in state.values()):
                    raise ValueError(f"检查点 {state_name} 包含无效或非有限模型参数")
            best = PolicyValueNet(config)
            best.load_state_dict(payload["best_model"], strict=True)
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=config.optimizer.learning_rate,
                                          weight_decay=config.optimizer.weight_decay)
            optimizer.load_state_dict(payload["optimizer"])
            if payload["training_steps"] and not optimizer.state:
                raise ValueError("检查点 optimizer 缺少已有训练步骤的动量状态")
            # Loading accepts malformed moment tensors that only fail on the
            # next update. Reject those before declaring the run recoverable.
            for parameter, state in optimizer.state.items():
                if not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                    raise ValueError("检查点 optimizer 缺少 AdamW 动量或步数")
                step = state["step"]
                if isinstance(step, torch.Tensor):
                    if step.numel() != 1 or not torch.isfinite(step).all() or step.item() < 0:
                        raise ValueError("检查点 optimizer 步数无效")
                elif not isinstance(step, (int, float)) or not math.isfinite(step) or step < 0:
                    raise ValueError("检查点 optimizer 步数无效")
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        if not torch.isfinite(value).all():
                            raise ValueError("检查点 optimizer 包含非有限状态")
                        if key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") and value.shape != parameter.shape:
                            raise ValueError("检查点 optimizer 动量形状与模型参数不一致")
            if payload["scaler"]:
                scale = payload["scaler"]
                for name in ("scale", "growth_factor", "backoff_factor"):
                    number = scale.get(name)
                    if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
                        raise ValueError(f"检查点 scaler.{name} 无效")
                if scale["growth_factor"] <= 1 or scale["backoff_factor"] >= 1:
                    raise ValueError("检查点 scaler 增长/退避系数无效")
                for name, minimum in (("growth_interval", 1), ("_growth_tracker", 0)):
                    if type(scale.get(name)) is not int or scale[name] < minimum:
                        raise ValueError(f"检查点 scaler.{name} 无效")
            scaler = torch.amp.GradScaler("cpu", enabled=bool(payload["scaler"]))
            scaler.load_state_dict(payload["scaler"])
            replay = ReplayBuffer(config.optimizer.replay_buffer_capacity,
                                  config.game.board_size,
                                  config.optimizer.use_board_symmetry_augmentation)
            replay.restore(payload["replay"])
            generator = np.random.default_rng()
            generator.bit_generator.state = deepcopy(payload["numpy_rng"])
            torch.Generator(device="cpu").set_state(payload["torch_rng"])
            cuda_rng = payload["cuda_rng"]
            if not isinstance(cuda_rng, (list, tuple)) or any(
                    not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
                    or state.ndim != 1 or not state.numel() for state in cuda_rng):
                raise ValueError("检查点 cuda_rng 必须是字节 RNG 状态列表")
            report = {"checkpoint": str(path.resolve()), "trainer": "native",
                      "checkpoint_loaded": True, "kind": "training",
                      "board_size": config.game.board_size,
                      "iteration": payload["iteration"],
                      "training_steps": payload["training_steps"],
                      "best_iteration": payload["best_iteration"],
                      "untrained_samples": payload.get("untrained_samples", 0),
                      "replay_samples": len(replay),
                      "model_sha256": fingerprint(payload["model"]),
                      "best_model_sha256": fingerprint(payload["best_model"]),
                      "optimizer_loaded": True, "scaler_loaded": True,
                      "rng_validated": True, "validation_device": "cpu"}
        return config, report
    except Exception as error:
        raise _error(f"普通版检查点验收失败：{path}：{error}") from error


def check_native_checkpoint(path: Path, expected_config: Path | None = None) -> dict:
    """Restricted-load all training state and validate it without writing files."""
    _config, report = _load_and_validate(Path(path), expected_config)
    return report


def native_source_plan(project_root: Path, run_dir: Path) -> dict:
    """Use authoritative saved configuration, never an optional sidecar."""
    root = (run_dir if run_dir.is_absolute() else project_root / run_dir).resolve()
    config, _report = _load_and_validate(root / "latest.pt")
    return {"kind": "single", "trainer": "native",
            "sources": [{"root": root, "prefix": "run", "board_size": config.game.board_size,
                         "lock": "run.lock"}],
            "configs": {"config/training.json": _config_document(config)},
            "active_prefix": "run"}
