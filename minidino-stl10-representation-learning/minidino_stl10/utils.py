import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mkdir(path: str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj: Dict[str, Any], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        return json.load(f)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_grad_scaler(device_type: str, enabled: bool):
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
):
    warmup_iters = warmup_epochs * steps_per_epoch
    total_iters = epochs * steps_per_epoch
    schedule = np.empty(total_iters, dtype=np.float64)

    if warmup_iters > 0:
        schedule[:warmup_iters] = np.linspace(start_warmup_value, base_value, warmup_iters)

    remaining_iters = total_iters - warmup_iters
    if remaining_iters > 0:
        iters = np.arange(remaining_iters)
        schedule[warmup_iters:] = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / remaining_iters)
        )
    return schedule


def set_optimizer_lr_wd(optimizer, lr: float, weight_decay: Optional[float] = None) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
        if weight_decay is not None and group.get("apply_weight_decay", True):
            group["weight_decay"] = weight_decay


def get_params_groups(model: torch.nn.Module):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name or "register_tokens" in name:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [
        {"params": regularized, "apply_weight_decay": True},
        {"params": not_regularized, "weight_decay": 0.0, "apply_weight_decay": False},
    ]


@torch.no_grad()
def update_teacher(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float) -> None:
    student_params = dict(student.named_parameters())
    teacher_params = dict(teacher.named_parameters())
    for name, teacher_param in teacher_params.items():
        student_param = student_params[name]
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

    student_buffers = dict(student.named_buffers())
    teacher_buffers = dict(teacher.named_buffers())
    for name, teacher_buffer in teacher_buffers.items():
        if name in student_buffers and teacher_buffer.dtype.is_floating_point:
            teacher_buffer.data.mul_(momentum).add_(student_buffers[name].data, alpha=1.0 - momentum)


def copy_student_to_teacher(student: torch.nn.Module, teacher: torch.nn.Module) -> None:
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False
