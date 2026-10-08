from __future__ import annotations

import numpy as np
import pytest

from tests.numsim.corpus.kernels.gemm import FP16_BF16_CONFIGS as NUMSIM_CONFIGS
from tests.numsim.corpus.kernels.gemm import _bfloat16_bits_to_float32, _float32_to_bfloat16_bits
from tests.numsim.corpus.kernels.gemm import prepare_fp16_bf16_case as prepare_numsim_case
from tests.numsim.microtests.harness import NUMSIM_GPU_MARK, require_numsim_gpu
from tests.numsim.support.three_way import run_three_way_case
from tests.numsim.support.kernel_facts import launch_topology


def test_gemm_numsim_corpus_matches_bootstrap_target():
    assert {config["dtype"] for config in NUMSIM_CONFIGS} == {"fp16", "bf16"}
    for config in NUMSIM_CONFIGS:
        assert (config["M"], config["N"], config["K"]) == (256, 2048, 64)


def test_independent_bfloat16_codec_rounds_to_nearest_even():
    values = np.array([1.0, -2.5, 1.00390625, 1.01171875], dtype=np.float32)
    encoded = _float32_to_bfloat16_bits(values)
    decoded = _bfloat16_bits_to_float32(encoded)

    np.testing.assert_array_equal(decoded, np.array([1.0, -2.5, 1.0, 1.015625], dtype=np.float32))


