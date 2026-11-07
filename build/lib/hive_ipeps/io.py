"""Portable real tensors and numerical output."""
import json
from pathlib import Path

import numpy as np
import torch

from .tensors import ANSATZE, DTYPE


def save_tensor(path, parameters, ansatz, J1=1., chi=6):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ansatz=ansatz, J1=J1, chi=chi,
             **{f"tensor_{i}": tensor.detach().cpu().numpy()
                for i, tensor in enumerate(parameters)})


def load_tensor(path, device="cpu"):
    with np.load(path, allow_pickle=False) as data:
        ansatz = str(data["ansatz"])
        if ansatz not in ANSATZE:
            raise ValueError("Unknown tensor ansatz.")
        count = {"3legSymmetrized1tensor": 1, "c3rotated2tensors": 2, "unrestricted6tensors": 6}[ansatz]
        parameters = [torch.tensor(data[f"tensor_{i}"], dtype=DTYPE, device=device)
                      for i in range(count)]
        return parameters, ansatz, float(data["J1"]), int(data["chi"])


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
