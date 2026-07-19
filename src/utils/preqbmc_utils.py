"""Compatibility wrapper for legacy utility imports."""

from __future__ import annotations

import random

import numpy as np

from synthesis.forward import forward_dnn, forward_dnn_multi
from utils.fixed_point import int_get_min_max, quantize_int, real_round


def forward_DNN(x, ilpModel):
    return forward_dnn(x, ilpModel)


def forward_DNN_multi(x_set, ilpModel):
    return forward_dnn_multi(x_set, ilpModel)


def backdoor_random(x_test, y_test, K, originalCls, targetCls):
    sample_list = [i for i in range(len(x_test))]
    sample_list = random.sample(sample_list, K * 100)
    real_sample_ID = []
    real_sample_input = []
    real_sample_label = []

    if originalCls < 10:
        for x_index in sample_list:
            if len(real_sample_input) >= K:
                break
            if y_test[x_index] == originalCls:
                real_sample_ID.append(x_index)
                real_sample_input.append(x_test[x_index])
                real_sample_label.append(y_test[x_index])
    else:
        for x_index in sample_list:
            if len(real_sample_input) >= K:
                break
            if y_test[x_index] != targetCls:
                real_sample_ID.append(x_index)
                real_sample_input.append(x_test[x_index])
                real_sample_label.append(y_test[x_index])

    assert len(real_sample_label) == K
    return real_sample_input, real_sample_ID, real_sample_label


__all__ = [
    "backdoor_random",
    "forward_DNN",
    "forward_DNN_multi",
    "int_get_min_max",
    "np",
    "quantize_int",
    "real_round",
]
