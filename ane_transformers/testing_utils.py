#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import numpy as np
import time
import torch


def rough_timeit(callable, n):
    assert n > 0
    tot = 0
    for _ in range(n):
        s = time.time()
        callable()
        tot += time.time() - s
    return tot / n


def compute_psnr(a, b):
    """ Compute Peak-Signal-to-Noise-Ratio across two numpy.ndarray objects
    """
    max_b = np.abs(b).max()
    sumdeltasq = 0.0

    sumdeltasq = ((a - b) * (a - b)).sum()

    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    eps = 1e-5
    eps2 = 1e-10
    psnr = 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))

    return psnr


def assert_rank(tensor, name, ranks):
    if tensor is not None:
        if isinstance(ranks, int):
            ranks = [ranks]
        assert isinstance(ranks, list)
        rank = len(tensor.shape)
        if rank not in ranks:
            raise ValueError(
                f"{name}'s rank is invalid (Expected one of {ranks}, got {rank})"
            )


def assert_shape(tensor, name, expected_shape):
    if tensor is not None:
        if isinstance(expected_shape, list):
            expected_shape = torch.Size(expected_shape)
        assert isinstance(expected_shape, torch.Size)
        shape = tensor.shape
        if shape != expected_shape:
            raise ValueError(
                f"{name}'s shape is invalid (Expected one of {expected_shape}, got {shape})"
            )
