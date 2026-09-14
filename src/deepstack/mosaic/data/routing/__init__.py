"""Routing inputs for the current reference models."""

from pathlib import Path

import numpy as np


def load_routing(model: str):
    if model not in {"qwen3_235b", "deepseek_v3"}:
        raise ValueError(f"unknown routing model: {model}")
    with np.load(Path(__file__).with_name(f"{model}.npz"), allow_pickle=False) as data:
        return data["prefill"], data["decode"]
