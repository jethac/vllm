# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Gemma 4 global-layer attention backend pin.

Under per-layer mixed KV (quantized cache_dtype + kv_cache_dtype_skip_layers)
the D>256 global layers must be pinned to TRITON_ATTN: the selector accepts
head_size=512 for FlashInfer but the FA2 kernel trait-guards it at run time.
The pin must apply identically to Gemma4Attention (target) and
Gemma4MTPAttention (assistant drafter), which reads the target's cache via
KV sharing.
"""

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.gemma4 import gemma4_global_attn_backend_override


def _cache_config(cache_dtype: str, skip_layers: list[str] | None):
    return SimpleNamespace(
        cache_dtype=cache_dtype,
        kv_cache_dtype_skip_layers=skip_layers,
    )


MIXED_KV = _cache_config("nvfp4", ["full_attention=auto"])


@pytest.fixture(autouse=True)
def _clear_vosplit_env(monkeypatch):
    monkeypatch.delenv("VLLM_FLASHINFER_VOSPLIT", raising=False)


def test_pin_applies_to_global_d512_under_mixed_kv():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    override = gemma4_global_attn_backend_override(
        MIXED_KV, is_sliding=False, head_dim=512
    )
    assert override is AttentionBackendEnum.TRITON_ATTN.get_class()


@pytest.mark.parametrize(
    ("cache_config", "is_sliding", "head_dim"),
    [
        # Sliding layers keep selector resolution (head 256 readable).
        (MIXED_KV, True, 256),
        # Homogeneous head dim: nothing to pin.
        (MIXED_KV, False, 256),
        # Full-NVFP4 (no skip layers): VO split route, no pin.
        (_cache_config("nvfp4", None), False, 512),
        # Unquantized cache: no mixed KV in play.
        (_cache_config("auto", ["full_attention=auto"]), False, 512),
        # No cache config at all (e.g. profiling paths).
        (None, False, 512),
    ],
)
def test_no_pin_outside_mixed_kv_global_layers(cache_config, is_sliding, head_dim):
    assert (
        gemma4_global_attn_backend_override(cache_config, is_sliding, head_dim)
        is None
    )


def test_all_dtype_vosplit_supersedes_pin(monkeypatch):
    monkeypatch.setenv("VLLM_FLASHINFER_VOSPLIT", "1")
    assert (
        gemma4_global_attn_backend_override(MIXED_KV, is_sliding=False, head_dim=512)
        is None
    )


def test_vosplit_zero_means_disabled(monkeypatch):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    monkeypatch.setenv("VLLM_FLASHINFER_VOSPLIT", "0")
    override = gemma4_global_attn_backend_override(
        MIXED_KV, is_sliding=False, head_dim=512
    )
    assert override is AttentionBackendEnum.TRITON_ATTN.get_class()


def test_target_and_drafter_use_the_same_helper():
    """Regression guard: Gemma4MTPAttention must resolve its backend with
    the same helper as Gemma4Attention so target and drafter never diverge
    per layer type again."""
    import inspect

    from vllm.model_executor.models import gemma4, gemma4_mtp

    target_src = inspect.getsource(gemma4.Gemma4Attention.__init__)
    drafter_src = inspect.getsource(gemma4_mtp.Gemma4MTPAttention.__init__)
    assert "gemma4_global_attn_backend_override" in target_src
    assert "gemma4_global_attn_backend_override" in drafter_src
