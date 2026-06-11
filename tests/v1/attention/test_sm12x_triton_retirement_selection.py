# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static (no-GPU) selection tests for the CC 12.x Triton retirement.

spark-hijinks campaign: VLLM_FLASHINFER_BF16_GEMMA routes Gemma-family
bf16-KV configs on consumer Blackwell (CC 12.x) to FlashInfer (with the
FA2 two-pass VO split for D=512 global layers), retiring the upstream
model-wide TRITON_ATTN force. These tests pin:

1. the model-config routing truth table (Gemma3Config / Gemma4Config),
2. FlashInferBackend.validate_configuration honesty for head_size > 256
   on CC 12.x (the banked selector-vs-kernel head-512 discrepancy), and
3. _vo_split_factor knob semantics.

Everything runs under a mocked platform/capability; no CUDA required.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    _vo_split_factor,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

KNOB = "VLLM_FLASHINFER_BF16_GEMMA"
ALL_KNOBS = (
    KNOB,
    "VLLM_FLASHINFER_VOSPLIT",
    "VLLM_NVFP4_KV_VOSPLIT",
    "VLLM_NVFP4_KV_LINEAR_V_SF",
    "VLLM_FLASHINFER_MM_PREFIX",
)

CC12_0 = DeviceCapability(12, 0)
CC12_1 = DeviceCapability(12, 1)
CC9_0 = DeviceCapability(9, 0)


@pytest.fixture(autouse=True)
def _clear_campaign_knobs(monkeypatch):
    for name in ALL_KNOBS:
        monkeypatch.delenv(name, raising=False)
    yield


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mock_vllm_config(
    *,
    backend=None,
    cache_dtype="auto",
    skip_layers=None,
    head_dim=256,
    global_head_dim=512,
    is_mm_prefix_lm=False,
):
    return SimpleNamespace(
        attention_config=SimpleNamespace(backend=backend, flash_attn_version=None),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            kv_cache_dtype_skip_layers=skip_layers,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                head_dim=head_dim, global_head_dim=global_head_dim
            ),
            is_mm_prefix_lm=is_mm_prefix_lm,
        ),
    )


@pytest.fixture
def fake_cc(monkeypatch):
    """Pin vllm.platforms.current_platform to a fake CUDA platform with a
    chosen compute capability, and pin is_fa_version_supported(4) to False
    (the CC 12.x reality: no FA4 wheel; keeps the upstream Gemma4 branch
    deterministic on any test host)."""

    def _set(capability: DeviceCapability):
        import vllm.platforms as platforms_mod
        import vllm.v1.attention.backends.fa_utils as fa_utils_mod

        fake = SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda device_id=0: capability,
        )
        monkeypatch.setattr(platforms_mod, "current_platform", fake, raising=False)
        monkeypatch.setattr(
            fa_utils_mod, "is_fa_version_supported", lambda v: False
        )
        return fake

    return _set


def _gemma3_route(vllm_config):
    from vllm.model_executor.models.config import Gemma3Config

    Gemma3Config.verify_and_update_config(vllm_config)
    return vllm_config.attention_config.backend


def _gemma4_route(vllm_config):
    from vllm.model_executor.models.config import Gemma4Config

    Gemma4Config.verify_and_update_config(vllm_config)
    return vllm_config.attention_config.backend


# ---------------------------------------------------------------------------
# 1. model-config routing truth table
# ---------------------------------------------------------------------------


class TestGemma4Routing:
    def test_knob_off_cc12_bf16_forces_triton(self, fake_cc):
        """Baseline upstream behavior on CC 12.x: no FA4 -> Triton force."""
        fake_cc(CC12_0)
        cfg = _mock_vllm_config()
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    @pytest.mark.parametrize("capability", [CC12_0, CC12_1])
    def test_knob_on_cc12_bf16_forces_flashinfer(
        self, fake_cc, monkeypatch, capability
    ):
        fake_cc(capability)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config()
        assert _gemma4_route(cfg) == AttentionBackendEnum.FLASHINFER

    def test_knob_on_explicit_bfloat16_cache_dtype_routes(
        self, fake_cc, monkeypatch
    ):
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config(cache_dtype="bfloat16")
        assert _gemma4_route(cfg) == AttentionBackendEnum.FLASHINFER

    def test_knob_on_uniform_head_dims_still_routes(self, fake_cc, monkeypatch):
        """The route sits before the heterogeneous-head early return."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config(head_dim=256, global_head_dim=256)
        assert _gemma4_route(cfg) == AttentionBackendEnum.FLASHINFER

    def test_knob_on_hopper_does_not_route(self, fake_cc, monkeypatch):
        """CC scope: the knob must not leak outside 12.x."""
        fake_cc(CC9_0)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config()
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    @pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3", "nvfp4"])
    def test_knob_on_quantized_kv_does_not_route(
        self, fake_cc, monkeypatch, cache_dtype
    ):
        """Dtype scope: quantized-KV configs keep their own routes."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config(cache_dtype=cache_dtype)
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    def test_knob_on_explicit_user_backend_wins(self, fake_cc, monkeypatch):
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config(backend=AttentionBackendEnum.TRITON_ATTN)
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    def test_knob_on_mm_prefix_lm_without_mm_knob_does_not_route(
        self, fake_cc, monkeypatch
    ):
        """Multimodal Gemma (mm-prefix spans live) cannot be forced onto
        FlashInfer without VLLM_FLASHINFER_MM_PREFIX: backend validation
        would hard-fail at startup. Upstream route must stand."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config(is_mm_prefix_lm=True)
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    def test_knob_on_mm_prefix_lm_with_mm_knob_routes(
        self, fake_cc, monkeypatch
    ):
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "1")
        monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", "1")
        cfg = _mock_vllm_config(is_mm_prefix_lm=True)
        assert _gemma4_route(cfg) == AttentionBackendEnum.FLASHINFER

    def test_existing_vosplit_knob_still_forces_flashinfer(
        self, fake_cc, monkeypatch
    ):
        """Regression: the pre-existing all-dtype VO-split route stands."""
        fake_cc(CC12_0)
        monkeypatch.setenv("VLLM_FLASHINFER_VOSPLIT", "1")
        cfg = _mock_vllm_config()
        assert _gemma4_route(cfg) == AttentionBackendEnum.FLASHINFER

    def test_existing_mixed_kv_route_keeps_per_layer_resolution(
        self, fake_cc, monkeypatch
    ):
        """Regression: mixed-KV (nvfp4 + skip layers) stays per-layer."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "1")  # must not interfere (dtype scope)
        cfg = _mock_vllm_config(cache_dtype="nvfp4", skip_layers="full_attention")
        assert _gemma4_route(cfg) is None

    def test_existing_nvfp4_vosplit_route_keeps_per_layer_resolution(
        self, fake_cc, monkeypatch
    ):
        fake_cc(CC12_0)
        monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "1")
        cfg = _mock_vllm_config(cache_dtype="nvfp4")
        assert _gemma4_route(cfg) is None


class TestGemma3Routing:
    def test_knob_off_leaves_backend_unset(self, fake_cc):
        fake_cc(CC12_0)
        cfg = _mock_vllm_config(global_head_dim=None)
        assert _gemma3_route(cfg) is None

    @pytest.mark.parametrize("capability", [CC12_0, CC12_1])
    def test_knob_on_cc12_bf16_forces_flashinfer(
        self, fake_cc, monkeypatch, capability
    ):
        fake_cc(capability)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config(global_head_dim=None)
        assert _gemma3_route(cfg) == AttentionBackendEnum.FLASHINFER

    def test_knob_on_hopper_does_not_route(self, fake_cc, monkeypatch):
        fake_cc(CC9_0)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config(global_head_dim=None)
        assert _gemma3_route(cfg) is None

    def test_knob_on_fp8_kv_does_not_route(self, fake_cc, monkeypatch):
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config(cache_dtype="fp8", global_head_dim=None)
        assert _gemma3_route(cfg) is None


# ---------------------------------------------------------------------------
# 2. FlashInfer selector honesty (selector-vs-kernel head-512 resolution)
# ---------------------------------------------------------------------------


def _validate(head_size, kv_cache_dtype, capability):
    return FlashInferBackend.validate_configuration(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=capability,
        attn_type="decoder",
    )


class TestFlashInferHead512SelectorHonesty:
    @pytest.mark.parametrize("capability", [CC12_0, CC12_1])
    @pytest.mark.parametrize("kv", ["auto", "bfloat16", "fp8"])
    def test_head256_always_valid(self, capability, kv):
        assert _validate(256, kv, capability) == []

    @pytest.mark.parametrize("capability", [CC12_0, CC12_1])
    @pytest.mark.parametrize("kv", ["auto", "bfloat16", "fp8"])
    def test_head512_rejected_without_knobs(self, capability, kv):
        """The resolved discrepancy: the FA2 kernel trait guard rejects
        HEAD_DIM_VO > 256 at runtime, so the selector must reject it at
        selection time when no VO-split knob is set."""
        reasons = _validate(512, kv, capability)
        assert any("VO split" in r or "VO-split" in r for r in reasons), reasons

    def test_head512_nvfp4_rejected_without_knobs(self):
        reasons = _validate(512, "nvfp4", CC12_1)
        assert any("VO split" in r or "VO-split" in r for r in reasons), reasons

    @pytest.mark.parametrize("capability", [CC12_0, CC12_1])
    @pytest.mark.parametrize("kv", ["auto", "bfloat16", "fp8"])
    def test_head512_valid_with_bf16_gemma_knob(self, monkeypatch, capability, kv):
        monkeypatch.setenv(KNOB, "1")
        assert _validate(512, kv, capability) == []

    @pytest.mark.parametrize("kv", ["auto", "fp8"])
    def test_head512_valid_with_vosplit_knob(self, monkeypatch, kv):
        monkeypatch.setenv("VLLM_FLASHINFER_VOSPLIT", "1")
        assert _validate(512, kv, CC12_1) == []

    def test_head512_nvfp4_not_enabled_by_bf16_knob(self, monkeypatch):
        """The bf16 knob must not promise the NVFP4 VO split (which
        additionally needs the linear V-SF cache layout)."""
        monkeypatch.setenv(KNOB, "1")
        reasons = _validate(512, "nvfp4", CC12_1)
        assert reasons != []

    def test_head512_nvfp4_needs_linear_v_sf(self, monkeypatch):
        monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "1")
        reasons = _validate(512, "nvfp4", CC12_1)
        assert any("LINEAR_V_SF" in r for r in reasons), reasons

    def test_head512_nvfp4_valid_with_full_knob_set(self, monkeypatch):
        monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "1")
        monkeypatch.setenv("VLLM_NVFP4_KV_LINEAR_V_SF", "1")
        assert _validate(512, "nvfp4", CC12_1) == []

    def test_non_cc12_behavior_untouched(self):
        """Scope guard: on non-12.x CCs the (documented) upstream
        over-promise stands — FlashInfer may route head 512 to TRTLLM
        kernels there, which we have not probed. No behavior change."""
        assert _validate(512, "auto", CC9_0) == []
        assert _validate(512, "auto", DeviceCapability(10, 0)) == []

    def test_triton_fallback_cell_is_reachable(self):
        """With FlashInfer honestly rejecting bf16 head-512 on CC 12.x
        (knobs off), per-layer automatic fallback must have a valid
        landing spot: Triton accepts the same cell."""
        triton_cls = AttentionBackendEnum.TRITON_ATTN.get_class()
        reasons = triton_cls.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=CC12_1,
            attn_type="decoder",
        )
        assert reasons == []


# ---------------------------------------------------------------------------
# 3. _vo_split_factor knob semantics
# ---------------------------------------------------------------------------


class TestVoSplitFactor:
    def test_head_le_256_never_splits(self, monkeypatch):
        monkeypatch.setenv(KNOB, "1")
        monkeypatch.setenv("VLLM_FLASHINFER_VOSPLIT", "1")
        assert _vo_split_factor(256, False) == 1
        assert _vo_split_factor(128, False) == 1

    def test_bf16_head512_no_knob_no_split(self):
        assert _vo_split_factor(512, False) == 1

    def test_bf16_head512_bf16_gemma_knob_splits(self, monkeypatch):
        monkeypatch.setenv(KNOB, "1")
        assert _vo_split_factor(512, False) == 2

    def test_bf16_head512_vosplit_knob_splits(self, monkeypatch):
        monkeypatch.setenv("VLLM_FLASHINFER_VOSPLIT", "1")
        assert _vo_split_factor(512, False) == 2

    def test_nvfp4_head512_no_knob_raises(self):
        with pytest.raises(ValueError, match="VO split"):
            _vo_split_factor(512, True)

    def test_nvfp4_head512_bf16_knob_does_not_enable(self, monkeypatch):
        monkeypatch.setenv(KNOB, "1")
        with pytest.raises(ValueError, match="VO split"):
            _vo_split_factor(512, True)

    def test_nvfp4_head512_full_knob_set_splits(self, monkeypatch):
        monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "1")
        monkeypatch.setenv("VLLM_NVFP4_KV_LINEAR_V_SF", "1")
        assert _vo_split_factor(512, True) == 2


# ---------------------------------------------------------------------------
# 4. envs.py knob registration
# ---------------------------------------------------------------------------


class TestEnvKnob:
    def test_default_off(self):
        import vllm.envs as envs

        assert envs.VLLM_FLASHINFER_BF16_GEMMA is False

    def test_set_on(self, monkeypatch):
        import vllm.envs as envs

        monkeypatch.setenv(KNOB, "1")
        assert envs.VLLM_FLASHINFER_BF16_GEMMA is True

    def test_zero_is_off(self, monkeypatch):
        import vllm.envs as envs

        monkeypatch.setenv(KNOB, "0")
        assert envs.VLLM_FLASHINFER_BF16_GEMMA is False
