# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS


def test_qwen3_5_text_models_registered():
    assert _TEXT_GENERATION_MODELS["Qwen3_5ForCausalLM"] == (
        "qwen3_5",
        "Qwen3_5ForCausalLM",
    )
    assert _TEXT_GENERATION_MODELS["Qwen3_5MoeForCausalLM"] == (
        "qwen3_5",
        "Qwen3_5MoeForCausalLM",
    )
