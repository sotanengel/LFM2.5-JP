"""Seed utility tests."""

import random

import numpy as np
import torch

from lfm25_ja.utils.seed import set_seed


def test_set_seed_reproducible() -> None:
    set_seed(123)
    a = [random.random(), np.random.rand(), torch.rand(1).item()]
    set_seed(123)
    b = [random.random(), np.random.rand(), torch.rand(1).item()]
    assert a == b


def test_different_seeds_differ() -> None:
    set_seed(1)
    x = random.random()
    set_seed(2)
    y = random.random()
    assert x != y
