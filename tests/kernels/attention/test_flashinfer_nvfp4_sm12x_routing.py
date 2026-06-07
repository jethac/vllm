# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytest.importorskip("flashinfer")

from vllm.v1.attention.backends import flashinfer as flashinfer_backend  # noqa: E402


class _FakeWrapper:
    def __init__(self, *args, backend=None, **kwargs):
        self.backend = backend


def _make_builder(*, use_fa2_nvfp4_kv: bool, is_kvcache_nvfp4: bool):
    builder = object.__new__(flashinfer_backend.FlashInferMetadataBuilder)
    builder._prefill_wrapper = None
    builder._decode_wrapper = None
    builder._decode_wrappers_cudagraph = {}
    builder.use_dcp = False
    builder.use_fa2_nvfp4_kv = use_fa2_nvfp4_kv
    builder.is_kvcache_nvfp4 = is_kvcache_nvfp4
    builder._get_workspace_buffer = lambda: None
    return builder


@pytest.mark.parametrize(
    ("use_fa2_nvfp4_kv", "is_kvcache_nvfp4", "expected_backend"),
    [
        (True, True, "fa2"),
        (False, True, "trtllm-gen"),
        (False, False, "auto"),
    ],
)
def test_flashinfer_prefill_wrapper_backend_for_nvfp4_sm12x(
    monkeypatch,
    use_fa2_nvfp4_kv,
    is_kvcache_nvfp4,
    expected_backend,
):
    monkeypatch.setattr(
        flashinfer_backend, "BatchPrefillWithPagedKVCacheWrapper", _FakeWrapper
    )
    monkeypatch.setattr(flashinfer_backend, "get_kv_cache_layout", lambda: "NHD")

    builder = _make_builder(
        use_fa2_nvfp4_kv=use_fa2_nvfp4_kv,
        is_kvcache_nvfp4=is_kvcache_nvfp4,
    )

    wrapper = builder._get_prefill_wrapper()

    assert wrapper.backend == expected_backend


@pytest.mark.parametrize(
    ("use_fa2_nvfp4_kv", "is_kvcache_nvfp4", "expected_backend"),
    [
        (True, True, "fa2"),
        (False, True, "trtllm-gen"),
        (False, False, "auto"),
    ],
)
def test_flashinfer_decode_wrapper_backend_for_nvfp4_sm12x(
    monkeypatch,
    use_fa2_nvfp4_kv,
    is_kvcache_nvfp4,
    expected_backend,
):
    monkeypatch.setattr(
        flashinfer_backend, "BatchDecodeWithPagedKVCacheWrapper", _FakeWrapper
    )
    monkeypatch.setattr(flashinfer_backend, "get_kv_cache_layout", lambda: "NHD")

    builder = _make_builder(
        use_fa2_nvfp4_kv=use_fa2_nvfp4_kv,
        is_kvcache_nvfp4=is_kvcache_nvfp4,
    )

    wrapper = builder._get_decode_wrapper(batch_size=1, use_cudagraph=False)

    assert wrapper.backend == expected_backend
