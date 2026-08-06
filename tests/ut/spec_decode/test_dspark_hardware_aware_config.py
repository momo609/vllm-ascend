# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import DSparkHardwareAwareVerificationConfig


def _vllm_config(dspark: bool = True) -> SimpleNamespace:
    return SimpleNamespace(speculative_config=SimpleNamespace(use_dspark=lambda: dspark))


def test_config_parses_measured_curves() -> None:
    config = DSparkHardwareAwareVerificationConfig(
        {
            "enabled": True,
            "draft_cost_curve": [[1, 0.2], [8, 0.8]],
            "verify_cost_curve": [[1, 0.5], [64, 1.5]],
        },
        _vllm_config(),
    )

    assert config.has_cost_curves
    assert config.draft_cost_curve == [(1, 0.2), (8, 0.8)]
    assert config.verify_cost_curve == [(1, 0.5), (64, 1.5)]


def test_config_rejects_non_dspark_method() -> None:
    with pytest.raises(ValueError, match="method='dspark'"):
        DSparkHardwareAwareVerificationConfig(
            {
                "enabled": True,
                "draft_cost_curve": [[1, 0.2]],
                "verify_cost_curve": [[1, 0.5]],
            },
            _vllm_config(dspark=False),
        )


@pytest.mark.parametrize(
    "curve",
    [
        [[0, 1.0]],
        [[1, -1.0]],
        [[2, 1.0], [1, 2.0]],
        [[1, "slow"]],
    ],
)
def test_config_rejects_invalid_cost_curve(curve) -> None:
    with pytest.raises(ValueError, match="draft_cost_curve"):
        DSparkHardwareAwareVerificationConfig(
            {"draft_cost_curve": curve},
            _vllm_config(),
        )
