import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mkdir(path: str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with Path(path).open('r') as f:
        return json.load(f)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: Optional[str] = 'cpu') -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_grad_scaler(device_type: str, enabled: bool):
    """Return a GradScaler with compatibility across PyTorch versions."""
    if not enabled:
        return None

    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler(device_type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)

    return torch.cuda.amp.GradScaler(enabled=enabled)


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if enabled and hasattr(torch, 'compile'):
        try:
            return torch.compile(model)
        except Exception as exc:
            print(f'torch.compile was requested but failed; continuing without it: {exc}')
    return model
