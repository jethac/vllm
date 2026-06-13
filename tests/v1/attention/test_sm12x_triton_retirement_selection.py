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

bf16 retirement is OPT-IN (2026-06-13, YAK-benchmark-driven). The
Amendment-3/4 default-on flips were REVERTED: the YAK Colab benchmark
measured Triton bf16 vs FlashInfer bf16 across all 10 Gemma 3+4 sizes and
found PARITY, not a speedup (Gemma 3 FA even carries slightly LESS KV
capacity). So changing the bf16 DEFAULT buys nothing and risks unmeasured
configs -- bf16 Gemma keeps the upstream backend (Triton for the 512-head
Gemma 4 geometry, FLASH_ATTN for Gemma 3) unless an explicit
VLLM_FLASHINFER_BF16_GEMMA=1. =0 is a master escape hatch (also stands down
the mm route). The earlier "FlashInfer wrong on sm_120 at d256/SWA-512"
claim was a refuted WSL2/WDDM artifact and is no longer load-bearing.

mm-prefix spans stay a DEFAULT-ON CAPABILITY route, DECOUPLED from the bf16
default: a multimodal Gemma model serves its bidirectional image spans on
the FlashInfer FA2 custom-mask path by default (VLLM_FLASHINFER_MM_PREFIX,
=0 to keep upstream) even when bf16 text routing is off -- because the
upstream backends can't serve those spans. nvfp4 KV (the actual feature)
routes via --kv-cache-dtype + VOSPLIT, independent of this function.
Backend-side cells (selector honesty, _vo_split_factor) are head>256 --
Gemma 4 geometry -- unchanged.

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
    def test_default_keeps_triton(self, fake_cc):
        """OPT-IN (YAK-benchmark revert): bf16 Triton == FlashInfer on
        throughput, so the bf16 DEFAULT is unchanged -- knob-unset text-only
        bf16 Gemma 4 keeps the upstream no-FA4 -> model-wide TRITON_ATTN
        force. =1 opts in to FlashInfer; nvfp4 routes via its own knobs."""
        fake_cc(CC12_0)
        cfg = _mock_vllm_config()
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    def test_knob_zero_cc12_bf16_restores_triton_force(self, fake_cc, monkeypatch):
        """Escape hatch: =0 restores the pre-flip upstream behavior on
        CC 12.x (no FA4 -> model-wide Triton force)."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "0")
        cfg = _mock_vllm_config()
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    def test_empty_string_behaves_like_unset(self, fake_cc, monkeypatch):
        """Opt-in parsing: the empty string is not an opt-in (like unset),
        so bf16 Gemma 4 keeps the upstream Triton force."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "")
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

    @pytest.mark.parametrize("knob", [None, "1"])
    def test_hopper_does_not_route(self, fake_cc, monkeypatch, knob):
        """CC scope: neither the default-on state nor an explicit =1 may
        leak outside 12.x."""
        fake_cc(CC9_0)
        if knob is not None:
            monkeypatch.setenv(KNOB, knob)
        cfg = _mock_vllm_config()
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    @pytest.mark.parametrize("knob", [None, "1"])
    @pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3", "nvfp4"])
    def test_quantized_kv_does_not_route(
        self, fake_cc, monkeypatch, cache_dtype, knob
    ):
        """Dtype scope: quantized-KV configs keep their own routes, both
        under the flipped default (knob unset) and explicit =1."""
        fake_cc(CC12_0)
        if knob is not None:
            monkeypatch.setenv(KNOB, knob)
        cfg = _mock_vllm_config(cache_dtype=cache_dtype)
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    @pytest.mark.parametrize("knob", [None, "1"])
    def test_explicit_user_backend_wins(self, fake_cc, monkeypatch, knob):
        """Explicit --attention-backend is never overridden, including by
        the flipped default."""
        fake_cc(CC12_0)
        if knob is not None:
            monkeypatch.setenv(KNOB, knob)
        cfg = _mock_vllm_config(backend=AttentionBackendEnum.TRITON_ATTN)
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    @pytest.mark.parametrize("knob", [None, "1"])
    def test_mm_prefix_lm_default_routes_flashinfer(
        self, fake_cc, monkeypatch, knob
    ):
        """FLIPPED by the Amendment 4 mm default flip (was: the mm
        carve-out kept the upstream Triton-capable route): multimodal
        Gemma 4 with VLLM_FLASHINFER_MM_PREFIX unset now routes to
        FLASHINFER by default — the spans run on the FA2 custom-mask
        path. Holds whether the bf16 routing is default-on (knob unset)
        or explicit =1."""
        fake_cc(CC12_0)
        if knob is not None:
            monkeypatch.setenv(KNOB, knob)
        cfg = _mock_vllm_config(is_mm_prefix_lm=True)
        assert _gemma4_route(cfg) == AttentionBackendEnum.FLASHINFER

    @pytest.mark.parametrize("knob", [None, "1"])
    def test_mm_prefix_lm_mm_knob_zero_keeps_triton(
        self, fake_cc, monkeypatch, knob
    ):
        """Escape hatch of the Amendment 4 flip: MM_PREFIX=0 stands the
        route down for mm models — FlashInfer cannot serve the spans, so
        the upstream (Triton-capable) route must stand, whether the bf16
        routing is default-on or explicit =1."""
        fake_cc(CC12_0)
        if knob is not None:
            monkeypatch.setenv(KNOB, knob)
        monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", "0")
        cfg = _mock_vllm_config(is_mm_prefix_lm=True)
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

    @pytest.mark.parametrize("knob", [None, "1"])
    def test_mm_prefix_lm_with_mm_knob_routes(self, fake_cc, monkeypatch, knob):
        """Explicit MM_PREFIX=1 routes exactly as pre-flip (both-knobs
        semantics), and now coincides with the default."""
        fake_cc(CC12_0)
        if knob is not None:
            monkeypatch.setenv(KNOB, knob)
        monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", "1")
        cfg = _mock_vllm_config(is_mm_prefix_lm=True)
        assert _gemma4_route(cfg) == AttentionBackendEnum.FLASHINFER

    def test_mm_prefix_lm_bf16_knob_zero_keeps_triton(self, fake_cc, monkeypatch):
        """The bf16 escape hatch reverts the WHOLE forced route for mm
        models too (mm spans then ride the upstream Gemma4 Triton force);
        VLLM_FLASHINFER_MM_PREFIX stays default-on but has no forced
        route to compose with."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "0")
        cfg = _mock_vllm_config(is_mm_prefix_lm=True)
        assert _gemma4_route(cfg) == AttentionBackendEnum.TRITON_ATTN

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
    @pytest.mark.parametrize("capability", [CC12_0, CC12_1])
    def test_default_leaves_backend_unset(self, fake_cc, capability):
        """OPT-IN (YAK-benchmark): bf16 Triton == FlashInfer (FA even slightly
        less KV capacity for Gemma 3), so the bf16 DEFAULT is unchanged.
        Knob-unset text-only Gemma 3 leaves the backend unset for upstream
        priority order (FLASH_ATTN where supported); only an explicit =1
        routes. (nvfp4 routes via its own knobs.)"""
        fake_cc(capability)
        cfg = _mock_vllm_config(global_head_dim=None)
        assert _gemma3_route(cfg) is None

    def test_empty_string_is_not_an_opt_in(self, fake_cc, monkeypatch):
        """Opt-in parsing: the empty string is not an opt-in, so Gemma 3
        bf16 stays on the upstream default."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "")
        cfg = _mock_vllm_config(global_head_dim=None)
        assert _gemma3_route(cfg) is None

    def test_knob_zero_leaves_backend_unset(self, fake_cc, monkeypatch):
        """=0 is the escape hatch: keeps upstream selection (backend
        unset). Post-re-flip this now DIFFERS from the knob-unset default
        (which routes FLASHINFER), matching the Gemma 4 cell."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "0")
        cfg = _mock_vllm_config(global_head_dim=None)
        assert _gemma3_route(cfg) is None

    @pytest.mark.parametrize("capability", [CC12_0, CC12_1])
    def test_knob_on_cc12_bf16_forces_flashinfer(
        self, fake_cc, monkeypatch, capability
    ):
        """Explicit =1 still opts Gemma 3 in (experiments only — the
        route is known numerically wrong on sm_120 d256/SWA-512; the
        routing code logs a warning to that effect)."""
        fake_cc(capability)
        monkeypatch.setenv(KNOB, "1")
        cfg = _mock_vllm_config(global_head_dim=None)
        assert _gemma3_route(cfg) == AttentionBackendEnum.FLASHINFER

    @pytest.mark.parametrize("knob", [None, "1"])
    def test_hopper_does_not_route(self, fake_cc, monkeypatch, knob):
        fake_cc(CC9_0)
        if knob is not None:
            monkeypatch.setenv(KNOB, knob)
        cfg = _mock_vllm_config(global_head_dim=None)
        assert _gemma3_route(cfg) is None

    @pytest.mark.parametrize("knob", [None, "1"])
    def test_fp8_kv_does_not_route(self, fake_cc, monkeypatch, knob):
        fake_cc(CC12_0)
        if knob is not None:
            monkeypatch.setenv(KNOB, knob)
        cfg = _mock_vllm_config(cache_dtype="fp8", global_head_dim=None)
        assert _gemma3_route(cfg) is None

    def test_mm_prefix_lm_default_routes_flashinfer(self, fake_cc):
        """Post-flip (2026-06-12 re-flip, scope-out refuted): Gemma 3 is
        now bf16 default_on=True, so the knob-unset early-return no longer
        fires; control reaches the mm-prefix branch, and with MM_PREFIX
        default-on (Amendment 4) a knob-unset Gemma 3 mm-prefix-lm routes
        FLASHINFER (the FA2 custom-mask span path). Set MM_PREFIX=0 to keep
        the Triton mm route -- pinned in
        test_mm_prefix_lm_mm_knob_zero_leaves_backend_unset."""
        fake_cc(CC12_0)
        cfg = _mock_vllm_config(global_head_dim=None, is_mm_prefix_lm=True)
        assert _gemma3_route(cfg) == AttentionBackendEnum.FLASHINFER

    @pytest.mark.parametrize("mm_knob", [None, "1"])
    def test_mm_prefix_lm_explicit_opt_in_routes_flashinfer(
        self, fake_cc, monkeypatch, mm_knob
    ):
        """Explicit Gemma 3 bf16 opt-in (=1) routes mm-prefix spans onto
        the FlashInfer FA2 custom-mask path (experiments only — known
        numerically wrong on sm_120 d256/SWA-512), whether MM_PREFIX is
        unset (default-on) or explicit =1; MM_PREFIX=0 stands it down."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "1")
        if mm_knob is not None:
            monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", mm_knob)
        cfg = _mock_vllm_config(global_head_dim=None, is_mm_prefix_lm=True)
        assert _gemma3_route(cfg) == AttentionBackendEnum.FLASHINFER

    def test_mm_prefix_lm_mm_knob_zero_leaves_backend_unset(
        self, fake_cc, monkeypatch
    ):
        """Escape hatch: MM_PREFIX=0 stands the route down; Gemma 3 mm
        falls back to the upstream priority order (backend unset)."""
        fake_cc(CC12_0)
        monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", "0")
        cfg = _mock_vllm_config(global_head_dim=None, is_mm_prefix_lm=True)
        assert _gemma3_route(cfg) is None

    def test_mm_prefix_lm_bf16_knob_zero_leaves_backend_unset(
        self, fake_cc, monkeypatch
    ):
        """The bf16 escape hatch reverts the whole forced route for mm
        Gemma 3 as well (upstream priority order decides; the mm default
        still lets FlashInfer CLAIM mm support there — pinned in
        TestFlashInferMMPrefixSupport)."""
        fake_cc(CC12_0)
        monkeypatch.setenv(KNOB, "0")
        cfg = _mock_vllm_config(global_head_dim=None, is_mm_prefix_lm=True)
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
    @pytest.mark.parametrize("kv", ["auto", "bfloat16"])
    def test_head512_bf16_valid_by_default(self, capability, kv):
        """FLIPPED by the Amendment 3 default flip: knob unset, bf16/auto
        KV head-512 on CC 12.x now validates (the default-on knob vouches
        for the FA2 two-pass VO split). Was: rejected without knobs."""
        assert _validate(512, kv, capability) == []

    @pytest.mark.parametrize("kv", ["auto", "bfloat16"])
    def test_head512_bf16_knob_zero_rejected(self, monkeypatch, kv):
        """Escape hatch: =0 restores the honest pre-flip rejection (the
        FA2 kernel trait guard rejects HEAD_DIM_VO > 256 at runtime, so
        with the split disabled the selector must reject at selection
        time)."""
        monkeypatch.setenv(KNOB, "0")
        reasons = _validate(512, kv, CC12_1)
        assert any("VO split" in r or "VO-split" in r for r in reasons), reasons

    @pytest.mark.parametrize("capability", [CC12_0, CC12_1])
    def test_head512_fp8_rejected_without_knobs(self, capability):
        """UNCHANGED by the default flip (fp8 routes untouched): the
        default-on knob only vouches for bf16/'auto' KV; fp8 head-512
        still needs an explicit VO-split opt-in."""
        reasons = _validate(512, "fp8", capability)
        assert any("VO split" in r or "VO-split" in r for r in reasons), reasons

    def test_head512_nvfp4_rejected_without_knobs(self):
        """UNCHANGED by the default flip (nvfp4 routes untouched)."""
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
        """Where FlashInfer honestly rejects head-512 on CC 12.x (fp8
        without explicit knobs, or bf16 with the =0 escape hatch),
        per-layer automatic fallback must have a valid landing spot:
        Triton accepts the same cell."""
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
# 2b. mm-prefix support scoping (Amendment 4 default flip)
# ---------------------------------------------------------------------------

GEMMA_MM_ARCHS = [
    ["Gemma3ForConditionalGeneration"],
    ["Gemma4ForConditionalGeneration"],
    ["Gemma4UnifiedForConditionalGeneration"],
]


@pytest.fixture
def fake_current_model(monkeypatch):
    """Pin vllm.config.get_current_vllm_config_or_none to a fake config
    holding a model with the given architectures. None = no vllm config
    in scope (a bare validate_configuration call): the mm default must
    then conservatively not vouch."""

    def _set(architectures):
        import vllm.config as config_mod

        if architectures is None:
            fake = lambda: None  # noqa: E731
        else:
            model_config = SimpleNamespace(architectures=architectures)
            fake = lambda: SimpleNamespace(model_config=model_config)  # noqa: E731
        monkeypatch.setattr(config_mod, "get_current_vllm_config_or_none", fake)

    return _set


class TestFlashInferMMPrefixSupport:
    @pytest.mark.parametrize("archs", GEMMA_MM_ARCHS)
    def test_default_claims_gemma_mm_on_cc12(
        self, fake_fi_cc_family, fake_current_model, archs
    ):
        """FLIPPED by the Amendment 4 mm default flip: knob unset,
        FlashInfer now claims mm-prefix support for Gemma 3/4 mm archs
        on CC 12.x devices (was: claim only with the knob set)."""
        fake_fi_cc_family(120)
        fake_current_model(archs)
        assert FlashInferBackend.supports_mm_prefix() is True

    def test_default_does_not_claim_non_gemma_archs(
        self, fake_fi_cc_family, fake_current_model
    ):
        """Arch scope: the default never vouches for mm-prefix archs
        (bagel/molmo2/moondream3/paligemma/umm) whose masking policy was
        not implemented/validated on this backend."""
        fake_fi_cc_family(120)
        fake_current_model(["Moondream3ForConditionalGeneration"])
        assert FlashInferBackend.supports_mm_prefix() is False

    def test_default_does_not_claim_off_cc12(
        self, fake_fi_cc_family, fake_current_model
    ):
        """CC scope: the default never leaks off 12.x."""
        fake_fi_cc_family(90)
        fake_current_model(["Gemma4ForConditionalGeneration"])
        assert FlashInferBackend.supports_mm_prefix() is False

    def test_default_without_config_in_scope_does_not_claim(
        self, fake_fi_cc_family, fake_current_model
    ):
        fake_fi_cc_family(120)
        fake_current_model(None)
        assert FlashInferBackend.supports_mm_prefix() is False

    def test_explicit_one_claims_unconditionally(
        self, monkeypatch, fake_fi_cc_family, fake_current_model
    ):
        """Explicit =1 keeps the pre-flip opt-in semantics: any CC, any
        mm-prefix arch."""
        fake_fi_cc_family(90)
        fake_current_model(["Moondream3ForConditionalGeneration"])
        monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", "1")
        assert FlashInferBackend.supports_mm_prefix() is True

    def test_zero_disables(
        self, monkeypatch, fake_fi_cc_family, fake_current_model
    ):
        """Escape hatch: =0 restores the pre-knob rejection even for
        Gemma mm archs on CC 12.x."""
        fake_fi_cc_family(120)
        fake_current_model(["Gemma4ForConditionalGeneration"])
        monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", "0")
        assert FlashInferBackend.supports_mm_prefix() is False


# ---------------------------------------------------------------------------
# 2c. mm x KV-dtype validation matrix (Amendment 4 interaction cells)
# ---------------------------------------------------------------------------


def _validate_mm(head_size, kv_cache_dtype, capability):
    return FlashInferBackend.validate_configuration(
        head_size=head_size,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=True,
        use_per_head_quant_scales=False,
        device_capability=capability,
        attn_type="decoder",
    )


class TestFlashInferMMKvDtypeMatrix:
    """validate_configuration cells with use_mm_prefix=True, pinning the
    mm flip x bf16 flip x quantized-KV interaction matrix. Default-mode
    cells mock a CC 12.x device family and a Gemma mm model in scope."""

    @pytest.fixture(autouse=True)
    def _gemma_on_cc12(self, fake_fi_cc_family, fake_current_model):
        self._set_model = fake_current_model
        fake_fi_cc_family(120)
        fake_current_model(["Gemma4ForConditionalGeneration"])

    @pytest.mark.parametrize("kv", ["auto", "bfloat16"])
    def test_mm_bf16_kv_accepted_by_default(self, kv):
        """FLIPPED by the Amendment 4 mm default flip (was: mm rejected
        without the knob)."""
        assert _validate_mm(256, kv, CC12_0) == []

    def test_mm_nvfp4_kv_accepted_by_default(self):
        """THE Amendment 4 interaction cell: mm spans with NVFP4 KV are
        probe-proven (nvfp4_d256 mask probe) and must route FlashInfer
        (head-256 sliding layers of Gemma 4, all layers of Gemma 3)."""
        assert _validate_mm(256, "nvfp4", CC12_0) == []

    def test_mm_nvfp4_head512_accepted_with_nvfp4_knobs(self, monkeypatch):
        """mm x NVFP4 x VO-split composition: Gemma 4 D512 global layers
        under full-NVFP4 KV still need the NVFP4 VO-split knob pair
        (unchanged); with it the mm default composes. (The 'vision'
        policy keeps these layers causal at build time, but selection is
        model-wide mm.)"""
        monkeypatch.setenv("VLLM_NVFP4_KV_VOSPLIT", "1")
        monkeypatch.setenv("VLLM_NVFP4_KV_LINEAR_V_SF", "1")
        assert _validate_mm(512, "nvfp4", CC12_0) == []

    def test_mm_bf16_head512_accepted_by_default(self):
        """mm flip x bf16 flip composition: both defaults vouch — bf16
        default for the head-512 VO split, mm default for the spans."""
        assert _validate_mm(512, "auto", CC12_0) == []

    def test_mm_fp8_kv_rejected_by_default(self):
        """KV-dtype scope of the mm default: fp8-KV mm spans were never
        probe-validated; the default must not vouch (upstream/Triton
        keeps the cell). Explicit =1 opts in."""
        reasons = _validate_mm(256, "fp8", CC12_0)
        assert any("MM_PREFIX" in r for r in reasons), reasons

    def test_mm_fp8_kv_accepted_with_explicit_knob(self, monkeypatch):
        """Explicit =1 keeps the pre-flip opt-in scope (any KV dtype)."""
        monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", "1")
        assert _validate_mm(256, "fp8", CC12_0) == []

    def test_mm_knob_zero_rejected(self, monkeypatch):
        """Escape hatch: =0 restores the pre-knob mm rejection."""
        monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", "0")
        reasons = _validate_mm(256, "auto", CC12_0)
        assert any("multimodal" in r for r in reasons), reasons

    def test_mm_non_gemma_arch_rejected_by_default(self):
        """Arch scope at the validation level."""
        self._set_model(["Moondream3ForConditionalGeneration"])
        reasons = _validate_mm(256, "auto", CC12_0)
        assert any("multimodal" in r for r in reasons), reasons

    def test_mm_fp8_triton_fallback_cell_is_reachable(self):
        """Where the mm default declines fp8 KV, the upstream fallback
        must have a valid landing spot: Triton accepts mm + fp8 KV."""
        triton_cls = AttentionBackendEnum.TRITON_ATTN.get_class()
        reasons = triton_cls.validate_configuration(
            head_size=256,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=True,
            use_per_head_quant_scales=False,
            device_capability=CC12_0,
            attn_type="decoder",
        )
        assert reasons == []


# ---------------------------------------------------------------------------
# 3. _vo_split_factor knob semantics
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_fi_cc_family(monkeypatch):
    """Pin the flashinfer module's current_platform to a fake whose
    device-capability family is the given one (e.g. 120 for CC 12.x).
    Needed because the DEFAULT-ON state of the bf16-Gemma knob is
    CC-scoped at the runtime sites (_vo_split_factor, cudagraph gate),
    where no DeviceCapability argument exists."""

    def _set(family: int):
        import vllm.v1.attention.backends.flashinfer as flashinfer_mod

        fake = SimpleNamespace(
            is_device_capability_family=lambda fam: fam == family,
        )
        monkeypatch.setattr(flashinfer_mod, "current_platform", fake)
        return fake

    return _set


class TestVoSplitFactor:
    def test_head_le_256_never_splits(self, monkeypatch):
        monkeypatch.setenv(KNOB, "1")
        monkeypatch.setenv("VLLM_FLASHINFER_VOSPLIT", "1")
        assert _vo_split_factor(256, False) == 1
        assert _vo_split_factor(128, False) == 1

    def test_bf16_head512_default_splits_on_cc12(self, fake_fi_cc_family):
        """FLIPPED by the Amendment 3 default flip: knob unset, the
        non-NVFP4 head-512 VO split engages by default on CC 12.x
        devices. Was: no knob -> no split."""
        fake_fi_cc_family(120)
        assert _vo_split_factor(512, False) == 2

    def test_bf16_head512_default_no_split_off_cc12(self, fake_fi_cc_family):
        """CC scope of the default: knob unset, non-12.x devices keep the
        pre-flip single-pass behavior."""
        fake_fi_cc_family(90)
        assert _vo_split_factor(512, False) == 1

    def test_bf16_head512_knob_zero_no_split(self, monkeypatch, fake_fi_cc_family):
        """Escape hatch: =0 disables the split even on CC 12.x."""
        fake_fi_cc_family(120)
        monkeypatch.setenv(KNOB, "0")
        assert _vo_split_factor(512, False) == 1

    def test_bf16_head512_bf16_gemma_knob_splits(
        self, monkeypatch, fake_fi_cc_family
    ):
        """Explicit =1 keeps the pre-flip opt-in semantics: enabled on
        any CC (mocked non-12.x here to pin that)."""
        fake_fi_cc_family(90)
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
    def test_default_on(self):
        """FLIPPED by the Amendment 3 default flip: the knob registers as
        default-on (was: default-off / opt-in)."""
        import vllm.envs as envs

        assert envs.VLLM_FLASHINFER_BF16_GEMMA is True

    def test_set_on(self, monkeypatch):
        import vllm.envs as envs

        monkeypatch.setenv(KNOB, "1")
        assert envs.VLLM_FLASHINFER_BF16_GEMMA is True

    def test_zero_is_off(self, monkeypatch):
        import vllm.envs as envs

        monkeypatch.setenv(KNOB, "0")
        assert envs.VLLM_FLASHINFER_BF16_GEMMA is False

    def test_mm_prefix_default_on(self):
        """FLIPPED by the Amendment 4 mm default flip: the mm knob
        registers as default-on (was: default-off / opt-in)."""
        import vllm.envs as envs

        assert envs.VLLM_FLASHINFER_MM_PREFIX is True

    def test_mm_prefix_set_on(self, monkeypatch):
        import vllm.envs as envs

        monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", "1")
        assert envs.VLLM_FLASHINFER_MM_PREFIX is True

    def test_mm_prefix_zero_is_off(self, monkeypatch):
        import vllm.envs as envs

        monkeypatch.setenv("VLLM_FLASHINFER_MM_PREFIX", "0")
        assert envs.VLLM_FLASHINFER_MM_PREFIX is False
