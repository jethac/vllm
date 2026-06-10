# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

from dataclasses import dataclass
from functools import partial
import json
import os
import sys
from typing import Any, ClassVar

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from flashinfer.decode import fast_decode_plan, trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import FP4Tensor
from typing_extensions import override

from vllm import envs
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    use_trtllm_attention,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.spark_tensor_trace import (
    spark_tensor_trace,
    spark_tensor_trace_should_emit,
    spark_trace_last_token_summary,
)
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
    is_strictly_contiguous,
    nvfp4_kv_cache_full_dim,
    nvfp4_kv_cache_split_views,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer

FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024
VLLM_NVFP4_V_SF_DESWIZZLE_FLAG = "-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1"

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

logger = init_logger(__name__)

trtllm_gen_workspace_buffer = None
_SPARK_KV_TRACE_COUNTS: dict[tuple[str, str], int] = {}
_SPARK_ACTIVE_PAGE_DUMP_COUNTS: dict[tuple[str, str], int] = {}


def _spark_kv_trace_enabled() -> bool:
    return os.environ.get("VLLM_SPARK_KV_TRACE") == "1"


def _spark_nvfp4_prefill_contig_out_enabled() -> bool:
    return os.environ.get("VLLM_SPARK_NVFP4_PREFILL_CONTIG_OUT") == "1"


def _spark_nvfp4_prefill_fresh_wrapper_replay_enabled() -> bool:
    return os.environ.get("VLLM_SPARK_NVFP4_PREFILL_FRESH_WRAPPER_REPLAY") == "1"


def _spark_kv_trace_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _spark_kv_trace_layer_name(layer: torch.nn.Module | None) -> str | None:
    if layer is None:
        return None
    name = getattr(layer, "layer_name", None)
    return str(name) if name is not None else None


def _spark_kv_trace_wants_layer(
    layer_name: str | None = None,
    layer_names: list[str] | None = None,
) -> bool:
    raw_filter = os.environ.get("VLLM_SPARK_KV_TRACE_LAYERS", "")
    filters = [item.strip() for item in raw_filter.split(",") if item.strip()]
    if not filters:
        return True
    candidates = []
    if layer_name is not None:
        candidates.append(layer_name)
    if layer_names is not None:
        candidates.extend(layer_names)
    return any(f in candidate for f in filters for candidate in candidates)


def _spark_kv_trace_should_emit(
    event: str,
    layer_name: str | None = None,
    layer_names: list[str] | None = None,
) -> bool:
    if not _spark_kv_trace_enabled():
        return False
    if not _spark_kv_trace_wants_layer(layer_name, layer_names):
        return False
    limit = _spark_kv_trace_int("VLLM_SPARK_KV_TRACE_LIMIT", 4)
    if limit == 0:
        return False
    layer_key = layer_name or ",".join(layer_names or ["<none>"])
    key = (event, layer_key)
    count = _SPARK_KV_TRACE_COUNTS.get(key, 0)
    if count >= limit:
        return False
    _SPARK_KV_TRACE_COUNTS[key] = count + 1
    return True


def _spark_kv_trace_tensor_head(
    tensor: torch.Tensor | None,
    limit: int | None = None,
) -> list[int | float | bool | str] | None:
    if tensor is None:
        return None
    if limit is None:
        limit = _spark_kv_trace_int("VLLM_SPARK_KV_TRACE_LIMIT", 4)
    if limit <= 0:
        return []
    tensor = tensor.detach()
    if tensor.numel() == 0:
        return []
    try:
        values: list[int | float | bool | str] = []
        shape = tuple(tensor.shape)
        for linear_idx in range(min(limit, tensor.numel())):
            if len(shape) == 0:
                index = ()
            else:
                index_parts = []
                remainder = linear_idx
                for size in reversed(shape):
                    index_parts.append(remainder % size)
                    remainder //= size
                index = tuple(reversed(index_parts))
            value = tensor[index].cpu().item()
            values.append(value)
        return values
    except Exception as exc:
        return [f"<read_error:{type(exc).__name__}>"]


def _spark_kv_trace_view_info(tensor: torch.Tensor | None) -> dict[str, object] | None:
    if tensor is None:
        return None
    payload: dict[str, object] = {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }
    try:
        payload["data_ptr"] = int(tensor.data_ptr())
        payload["storage_data_ptr"] = int(tensor.untyped_storage().data_ptr())
    except Exception as exc:
        payload["ptr_error"] = type(exc).__name__
    return payload


def _spark_kv_trace_views_info(
    views: tuple[torch.Tensor, ...] | None,
) -> list[dict[str, object] | None] | None:
    if views is None:
        return None
    return [_spark_kv_trace_view_info(view) for view in views]


def _spark_trace_scalar(value: object) -> float | int | str | None:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return f"<tensor:{list(value.shape)}>"
            return float(value.detach().cpu().item())
        if isinstance(value, (float, int)):
            return value
        return float(value)  # type: ignore[arg-type]
    except Exception as exc:
        return f"<scalar_error:{type(exc).__name__}>"


def _spark_tensor_trace_view_payload(
    tensor: torch.Tensor | None,
) -> dict[str, object] | None:
    if tensor is None:
        return None
    head_tensor = tensor
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        head_tensor = tensor.view(torch.uint8)
    return {
        "view": _spark_kv_trace_view_info(tensor),
        "head": _spark_kv_trace_tensor_head(head_tensor),
    }


def _spark_tensor_trace_tuple_payload(
    tensors: tuple[torch.Tensor, ...] | torch.Tensor | None,
) -> list[dict[str, object] | None] | dict[str, object] | None:
    if tensors is None:
        return None
    if isinstance(tensors, tuple):
        return [_spark_tensor_trace_view_payload(tensor) for tensor in tensors]
    return _spark_tensor_trace_view_payload(tensors)


def _spark_tensor_trace_compare_payload(
    lhs: torch.Tensor | None,
    rhs: torch.Tensor | None,
) -> dict[str, object] | None:
    if lhs is None or rhs is None:
        return None
    if lhs.shape != rhs.shape:
        return {
            "lhs_shape": list(lhs.shape),
            "rhs_shape": list(rhs.shape),
            "shape_mismatch": True,
        }
    try:
        lhs_f = lhs.detach().float().reshape(-1)
        rhs_f = rhs.detach().float().reshape(-1)
        diff = lhs_f - rhs_f
        finite = torch.isfinite(lhs_f) & torch.isfinite(rhs_f)
        payload: dict[str, object] = {
            "shape": list(lhs.shape),
            "finite": int(finite.sum().cpu().item()),
        }
        if bool(finite.any().cpu().item()):
            diff_f = diff[finite]
            payload.update(
                {
                    "max_abs_diff": float(diff_f.abs().max().cpu().item()),
                    "mean_abs_diff": float(diff_f.abs().mean().cpu().item()),
                    "rms_diff": float(
                        torch.sqrt(torch.mean(diff_f * diff_f)).cpu().item()
                    ),
                }
            )
            lhs_norm = torch.linalg.vector_norm(lhs_f[finite])
            rhs_norm = torch.linalg.vector_norm(rhs_f[finite])
            denom = lhs_norm * rhs_norm
            if bool((denom > 0).cpu().item()):
                payload["cosine"] = float(
                    (torch.dot(lhs_f[finite], rhs_f[finite]) / denom)
                    .cpu()
                    .item()
                )
        return payload
    except Exception as exc:
        return {"compare_error": type(exc).__name__}


def _spark_safe_repr(value: object, limit: int = 240) -> str | None:
    if value is None:
        return None
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr_error:{type(exc).__name__}>"
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _spark_trace_tensor_buffer_payload(
    tensor: torch.Tensor | None,
) -> dict[str, object] | None:
    if tensor is None:
        return None
    flat = tensor.detach().reshape(-1)
    values = _spark_kv_trace_int("VLLM_SPARK_GEMMA_TENSOR_TRACE_VALUES", 16)
    payload = _spark_tensor_trace_view_payload(tensor) or {}
    if values > 0 and flat.numel() > values:
        try:
            payload["tail"] = _spark_kv_trace_tensor_head(flat[-values:], values)
        except Exception as exc:
            payload["tail_error"] = type(exc).__name__
    return payload


def _spark_trace_module_payload(module: object) -> dict[str, object] | None:
    if module is None:
        return None
    return {
        "type": type(module).__name__,
        "name": getattr(module, "__name__", None),
        "file": getattr(module, "__file__", None),
        "repr": _spark_safe_repr(module),
        "has_plan": hasattr(module, "plan"),
        "has_paged_run": hasattr(module, "paged_run"),
        "has_ragged_run": hasattr(module, "ragged_run"),
    }


def _spark_trace_prefill_wrapper_payload(
    wrapper: BatchPrefillWithPagedKVCacheWrapper,
) -> dict[str, object]:
    flashinfer_module = sys.modules.get("flashinfer")
    flashinfer_prefill_module = sys.modules.get("flashinfer.prefill")
    return {
        "wrapper_type": type(wrapper).__name__,
        "flashinfer_file": getattr(flashinfer_module, "__file__", None),
        "flashinfer_prefill_file": getattr(
            flashinfer_prefill_module, "__file__", None
        ),
        "flashinfer_extra_cudaflags": os.environ.get(
            "FLASHINFER_EXTRA_CUDAFLAGS", ""
        ),
        "backend": getattr(wrapper, "_backend", None),
        "kv_layout": getattr(wrapper, "_kv_layout", None),
        "use_cuda_graph": bool(getattr(wrapper, "_use_cuda_graph", False)),
        "causal": bool(getattr(wrapper, "_causal", False)),
        "window_left": int(getattr(wrapper, "_window_left", -9999)),
        "logits_soft_cap": _spark_trace_scalar(
            getattr(wrapper, "_logits_soft_cap", None)
        ),
        "sm_scale": _spark_trace_scalar(getattr(wrapper, "_sm_scale", None)),
        "batch_size": int(getattr(wrapper, "_batch_size", -1)),
        "num_qo_heads": int(getattr(wrapper, "_num_qo_heads", -1)),
        "num_kv_heads": int(getattr(wrapper, "_num_kv_heads", -1)),
        "qo_indptr_last": int(getattr(wrapper, "_qo_indptr_last", -1)),
        "max_q_len": int(getattr(wrapper, "_max_q_len", -1)),
        "max_kv_len": int(getattr(wrapper, "_max_kv_len", -1)),
        "workspace_size": int(getattr(wrapper, "_workspace_size", -1)),
        "cached_q_data_type": str(getattr(wrapper, "_cached_q_data_type", None)),
        "cached_kv_data_type": str(getattr(wrapper, "_cached_kv_data_type", None)),
        "cached_o_data_type": str(getattr(wrapper, "_cached_o_data_type", None)),
        "fixed_split_size": int(
            getattr(wrapper, "vllm_prefill_fixed_split_size", -9999)
        ),
        "disable_split_kv": bool(getattr(wrapper, "vllm_disable_split_kv", False)),
        "plan_info_type": type(getattr(wrapper, "_plan_info", None)).__name__,
        "plan_info_repr": _spark_safe_repr(getattr(wrapper, "_plan_info", None)),
        "cached_module": _spark_trace_module_payload(
            getattr(wrapper, "_cached_module", None)
        ),
        "jit_module": _spark_trace_module_payload(getattr(wrapper, "_jit_module", None)),
        "qo_indptr": _spark_trace_tensor_buffer_payload(
            getattr(wrapper, "_qo_indptr_buf", None)
        ),
        "paged_kv_indptr": _spark_trace_tensor_buffer_payload(
            getattr(wrapper, "_paged_kv_indptr_buf", None)
        ),
        "paged_kv_indices": _spark_trace_tensor_buffer_payload(
            getattr(wrapper, "_paged_kv_indices_buf", None)
        ),
        "paged_kv_last_page_len": _spark_trace_tensor_buffer_payload(
            getattr(wrapper, "_paged_kv_last_page_len_buf", None)
        ),
    }


def _spark_active_page_dump_enabled() -> bool:
    return os.environ.get("VLLM_SPARK_ACTIVE_PAGE_DUMP") == "1"


def _spark_active_page_dump_dir() -> str:
    return os.environ.get("VLLM_SPARK_ACTIVE_PAGE_DUMP_DIR", "/tmp")


def _spark_active_page_dump_limit() -> int:
    return _spark_kv_trace_int("VLLM_SPARK_ACTIVE_PAGE_DUMP_LIMIT", 4)


def _spark_active_page_dump_pages() -> int:
    return _spark_kv_trace_int("VLLM_SPARK_ACTIVE_PAGE_DUMP_PAGES", 8)


def _spark_active_page_dump_should_emit(event: str, layer_name: str | None) -> bool:
    if not _spark_active_page_dump_enabled():
        return False
    if not _spark_kv_trace_wants_layer(layer_name):
        return False
    limit = _spark_active_page_dump_limit()
    if limit == 0:
        return False
    key = (event, layer_name or "<none>")
    count = _SPARK_ACTIVE_PAGE_DUMP_COUNTS.get(key, 0)
    if count >= limit:
        return False
    _SPARK_ACTIVE_PAGE_DUMP_COUNTS[key] = count + 1
    return True


def _spark_active_page_dump_path(
    event: str,
    layer_name: str | None,
    count: int,
) -> str:
    safe_layer = (layer_name or "unknown").replace("/", "_").replace(".", "_")
    safe_layer = safe_layer[-160:]
    return os.path.join(
        _spark_active_page_dump_dir(),
        f"spark_active_page_{event}_{safe_layer}_{count:04d}.pt",
    )


def _spark_active_page_dump_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu()


def _spark_active_page_dump(
    *,
    event: str,
    layer_name: str | None,
    query: torch.Tensor,
    out_before: torch.Tensor,
    out_after: torch.Tensor | None,
    kv_data: tuple[torch.Tensor, ...] | None,
    kv_scales: tuple[torch.Tensor, ...] | None,
    wrapper: BatchPrefillWithPagedKVCacheWrapper,
    k_scale: object,
    v_scale: object,
    window_left: int,
    num_prefill_tokens: int,
    num_decode_tokens: int,
) -> None:
    if not _spark_active_page_dump_should_emit(event, layer_name):
        return
    if kv_data is None or kv_scales is None:
        return

    count = _SPARK_ACTIVE_PAGE_DUMP_COUNTS[(event, layer_name or "<none>")]
    max_pages = _spark_active_page_dump_pages()
    try:
        os.makedirs(_spark_active_page_dump_dir(), exist_ok=True)
        indptr = getattr(wrapper, "_paged_kv_indptr_buf", None)
        indices = getattr(wrapper, "_paged_kv_indices_buf", None)
        last_page_len = getattr(wrapper, "_paged_kv_last_page_len_buf", None)
        if indptr is None or indices is None or last_page_len is None:
            return

        indptr_cpu = indptr.detach().cpu()
        last_page_len_cpu = last_page_len.detach().cpu()
        num_indices = int(indptr_cpu[-1].item()) if indptr_cpu.numel() else 0
        indices_cpu = indices[:num_indices].detach().cpu()
        active_pages = torch.unique(indices_cpu).to(torch.long)
        if max_pages > 0:
            active_pages = active_pages[:max_pages]

        payload: dict[str, object] = {
            "schema": "spark-active-page-prefill-dump/v1",
            "event": event,
            "layer_name": layer_name,
            "window_left": int(window_left),
            "num_prefill_tokens": int(num_prefill_tokens),
            "num_decode_tokens": int(num_decode_tokens),
            "k_scale": _spark_trace_scalar(k_scale),
            "v_scale": _spark_trace_scalar(v_scale),
            "query": _spark_active_page_dump_tensor(query),
            "out_before": _spark_active_page_dump_tensor(out_before),
            "out_after": _spark_active_page_dump_tensor(out_after),
            "paged_kv_indptr": indptr_cpu,
            "paged_kv_indices": indices_cpu,
            "paged_kv_last_page_len": last_page_len_cpu,
            "active_pages": active_pages,
            "kv_data_views": _spark_kv_trace_views_info(kv_data),
            "kv_scale_views": _spark_kv_trace_views_info(kv_scales),
            "kv_data_pages": tuple(
                _spark_active_page_dump_tensor(view[active_pages.to(view.device)])
                for view in kv_data
            ),
            "kv_scale_pages": tuple(
                _spark_active_page_dump_tensor(view[active_pages.to(view.device)])
                for view in kv_scales
            ),
        }
        torch.save(payload, _spark_active_page_dump_path(event, layer_name, count))
    except Exception:
        logger.exception(
            "Failed to dump Spark active FlashInfer prefill pages for %s",
            layer_name,
        )


def _spark_kv_trace_slot_samples(
    data_views: tuple[torch.Tensor, ...] | None,
    scale_views: tuple[torch.Tensor, ...] | None,
    slot_mapping: torch.Tensor | None,
    page_size: int,
) -> list[dict[str, object]]:
    if data_views is None or scale_views is None or slot_mapping is None:
        return []
    num_slots = _spark_kv_trace_int("VLLM_SPARK_KV_TRACE_LIMIT", 4)
    num_values = _spark_kv_trace_int("VLLM_SPARK_KV_TRACE_VALUES", 8)
    if num_slots <= 0 or num_values <= 0:
        return []
    slots = _spark_kv_trace_tensor_head(slot_mapping, num_slots) or []
    samples: list[dict[str, object]] = []
    for raw_slot in slots:
        try:
            slot = int(raw_slot)
            page = slot // page_size
            offset = slot % page_size
            sample: dict[str, object] = {
                "slot": slot,
                "page": page,
                "offset": offset,
            }
            for prefix, views in (("data", data_views), ("scale", scale_views)):
                for side, view in zip(("k", "v"), views):
                    row = view[page, offset, 0]
                    if prefix == "scale":
                        row = row.view(torch.uint8)
                    sample[f"{side}_{prefix}_head"] = _spark_kv_trace_tensor_head(
                        row, num_values
                    )
            samples.append(sample)
        except Exception as exc:
            samples.append(
                {
                    "slot": raw_slot,
                    "error": type(exc).__name__,
                }
            )
    return samples


def _spark_kv_trace(event: str, payload: dict[str, object]) -> None:
    if not _spark_kv_trace_enabled():
        return
    record = {
        "event": event,
        "pid": os.getpid(),
        **payload,
    }
    try:
        line = json.dumps(record, sort_keys=True, default=str)
    except Exception as exc:
        line = json.dumps(
            {
                "event": event,
                "pid": os.getpid(),
                "json_error": type(exc).__name__,
            },
            sort_keys=True,
        )
    path = os.environ.get("VLLM_SPARK_KV_TRACE_FILE")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as trace_file:
                trace_file.write(line + "\n")
            return
        except OSError as exc:
            logger.warning("Failed to write Spark KV trace %s: %s", path, exc)
    logger.warning("Spark KV trace: %s", line)


def _vllm_nvfp4_linear_v_sf() -> bool:
    """One knob couples writer and reader V-scale-factor layouts.

    VLLM_NVFP4_KV_LINEAR_V_SF=1 makes the NVFP4 cache writer store V scale
    factors linearly (same layout as K; read by
    reshape_and_cache_nvfp4_dispatch in C++) and makes the FlashInfer FA2
    reader consume them without the in-kernel de-swizzle. Linear V-SF is
    required for head-dim-sliced V views (Gemma 4 D=512 VO-split) because
    the trtllm 4-token swizzle does not commute with head-dim slicing.
    """
    value = os.environ.get("VLLM_NVFP4_KV_LINEAR_V_SF", "")
    return value not in ("", "0")


def _ensure_vllm_nvfp4_kv_deswizzle_flag() -> None:
    if _vllm_nvfp4_linear_v_sf():
        # Writer stores V-SF linearly; the reader must NOT de-swizzle.
        logger.info_once(
            "VLLM_NVFP4_KV_LINEAR_V_SF=1: NVFP4 V scale factors are linear; "
            "FlashInfer in-kernel V-SF de-swizzle disabled."
        )
        return
    extra_flags = os.environ.get("FLASHINFER_EXTRA_CUDAFLAGS", "")
    if "FLASHINFER_PAGED_V_SF_DESWIZZLE" in extra_flags:
        return
    os.environ["FLASHINFER_EXTRA_CUDAFLAGS"] = (
        f"{extra_flags} {VLLM_NVFP4_V_SF_DESWIZZLE_FLAG}".strip()
    )


def _vllm_nvfp4_kv_vosplit_requested() -> bool:
    """VLLM_NVFP4_KV_VOSPLIT=1 opts head_size > 256 NVFP4 layers into the
    FA2 two-pass VO split (Gemma 4 global D=512 layers)."""
    value = os.environ.get("VLLM_NVFP4_KV_VOSPLIT", "")
    return value not in ("", "0")


def _vllm_flashinfer_vosplit_requested() -> bool:
    """VLLM_FLASHINFER_VOSPLIT=1 opts head_size > 256 layers into the FA2
    two-pass VO split for ALL KV dtypes (bf16/fp8/NVFP4). This is the
    wholesale alternative to the Gemma 4 model-wide TRITON_ATTN force
    (cf. vllm-project/vllm#38887, #40677)."""
    value = os.environ.get("VLLM_FLASHINFER_VOSPLIT", "")
    return value not in ("", "0")


def _vo_split_factor(head_size: int, is_fa2_nvfp4: bool) -> int:
    """Number of VO passes for the FlashInfer FA2 path.

    The FA2 kernel trait guard rejects HEAD_DIM_VO > 256 (the per-thread
    output-accumulator fragments do not fit the register budget), but
    HEAD_DIM_QK=512 is fine, and attention decomposes exactly along the VO
    dimension: S = Q @ K^T and the softmax are identical per pass, and
    O = [P @ V_left | P @ V_right] concatenates with no LSE merge. So
    head_size 512 runs as two (head_dim_qk=512, head_dim_vo=256) passes
    over zero-copy V half views. Dtype-independent (the guard counts only
    accumulator fragments); NVFP4 additionally needs the linear V-SF
    layout so the scale factors slice along the head dim.
    """
    if head_size <= 256:
        return 1
    vosplit_all_dtypes = _vllm_flashinfer_vosplit_requested()
    if is_fa2_nvfp4:
        if not (vosplit_all_dtypes or _vllm_nvfp4_kv_vosplit_requested()):
            raise ValueError(
                f"NVFP4 KV with head_size={head_size} on the SM12x FA2 path "
                "needs the two-pass VO split (the FA2 kernel caps "
                "HEAD_DIM_VO at 256). Set VLLM_NVFP4_KV_VOSPLIT=1 and "
                "VLLM_NVFP4_KV_LINEAR_V_SF=1 to enable it, or keep these "
                "layers on a different KV dtype via "
                "--kv-cache-dtype-skip-layers."
            )
        if not _vllm_nvfp4_linear_v_sf():
            raise ValueError(
                "The NVFP4 VO split requires VLLM_NVFP4_KV_LINEAR_V_SF=1: "
                "the swizzled V-scale-factor layout spreads each 4-token "
                "group across the full scale row and cannot be sliced "
                "along the head dimension."
            )
    elif not vosplit_all_dtypes:
        # Dense/fp8 KV: without the knob, keep the pre-existing behavior
        # (backend selection / the Gemma4 TRITON_ATTN force handles >256).
        return 1
    split = -(-head_size // 256)
    if head_size % split != 0 or (
        is_fa2_nvfp4 and (head_size // split) % 16 != 0
    ):
        raise ValueError(
            "The VO split needs head_size divisible into <=256-wide "
            f"chunks{' of whole 16-element scale blocks' if is_fa2_nvfp4 else ''};"
            f" got head_size={head_size}."
        )
    return split


def _flashinfer_dtype_uri_part(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "").replace(".", "_")


def _fa2_nvfp4_prefill_jit_args(
    *,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
    o_data_type: torch.dtype,
    idtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    pos_encoding_mode: int = 0,
    use_fp16_qk_reduction: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """Build a FlashInfer FA2 paged-prefill JIT module that declares FP4 KV."""

    uri = (
        "vllm_batch_prefill_nvfp4_kv_"
        f"dtype_q_{_flashinfer_dtype_uri_part(q_data_type)}_"
        "dtype_kv_fp4x2_e2m1_"
        f"dtype_o_{_flashinfer_dtype_uri_part(o_data_type)}_"
        f"dtype_idx_{_flashinfer_dtype_uri_part(idtype)}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"swa_{int(use_sliding_window)}_"
        f"logits_cap_{int(use_logits_soft_cap)}_"
        f"fp16_qk_{int(use_fp16_qk_reduction)}"
    )
    jit_args: list[Any] = [
        uri,
        q_data_type,
        kv_data_type,
        o_data_type,
        idtype,
        head_dim_qk,
        head_dim_vo,
        [
            "maybe_custom_mask",
            "maybe_mask_indptr",
            "maybe_alibi_slopes",
            "maybe_prefix_len_ptr",
            "maybe_token_pos_in_items_ptr",
            "maybe_max_item_len_ptr",
            "maybe_k_cache_sf",
            "maybe_v_cache_sf",
        ],
        [
            "uint8_t",
            "int32_t",
            "float",
            "uint32_t",
            "uint16_t",
            "uint16_t",
            "uint8_t",
            "uint8_t",
        ],
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
            "token_pos_in_items_len",
        ],
        ["double", "double", "double", "double", "int64_t"],
        (
            "DefaultAttention<use_custom_mask, "
            f"{str(use_sliding_window).lower()}, "
            f"{str(use_logits_soft_cap).lower()}, "
            f"{str(pos_encoding_mode == 2).lower()}>"
        ),
        "#include<flashinfer/attention/variants.cuh>",
    ]
    jit_kwargs = {
        "pos_encoding_mode": pos_encoding_mode,
        "use_sliding_window": use_sliding_window,
        "use_logits_soft_cap": use_logits_soft_cap,
        "use_fp16_qk_reduction": use_fp16_qk_reduction,
        "fp8_enabled": False,
    }
    return jit_args, jit_kwargs


def _get_trtllm_gen_workspace_buffer():
    global trtllm_gen_workspace_buffer
    if trtllm_gen_workspace_buffer is None:
        trtllm_gen_workspace_buffer = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"
        )
    return trtllm_gen_workspace_buffer


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    src_stride_page,
    src_stride_kv,
    src_stride_head,
    DST_K_CACHE_STRIDE: tl.constexpr,
    DST_KV_CACHE_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    mock_page_idx = batch_idx * block_table_stride + mock_block_table_idx + 1
    head_offsets = tl.arange(0, HEAD_STRIDE)

    for h in range(NUM_KV_HEADS):
        h_off = tl.cast(h, tl.int64)

        # Read K from source (supports non-contiguous page/kv/head strides)
        src_k = orig_page_num * src_stride_page + h_off * src_stride_head + head_offsets
        fp8_k = tl.load(kv_cache_ptr + src_k)
        dequant_k = (fp8_k.to(tl.float32) * k_scale_val).to(dequant_dtype)

        # Write K to contiguous mock cache
        dst_k = mock_page_idx * DST_KV_CACHE_STRIDE + h * HEAD_STRIDE + head_offsets
        tl.store(mock_kv_cache_ptr + dst_k, dequant_k)

        # Read V from source (offset by src_stride_kv for the V half)
        src_v = (
            orig_page_num * src_stride_page
            + src_stride_kv
            + h_off * src_stride_head
            + head_offsets
        )
        fp8_v = tl.load(kv_cache_ptr + src_v)
        dequant_v = (fp8_v.to(tl.float32) * v_scale_val).to(dequant_dtype)

        # Write V to contiguous mock cache
        dst_v = (
            mock_page_idx * DST_KV_CACHE_STRIDE
            + DST_K_CACHE_STRIDE
            + h * HEAD_STRIDE
            + head_offsets
        )
        tl.store(mock_kv_cache_ptr + dst_v, dequant_v)


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)

    num_kv_heads, block_size, head_size = s[2], s[3], s[4]
    head_stride = block_size * head_size
    k_cache_stride = num_kv_heads * head_stride
    kv_cache_stride = k_cache_stride * s[1]

    strides = kv_cache.stride()
    assert strides[3] == head_size and strides[4] == 1, (
        "For kv cache layouts, (block_size, head_size) "
        f"dimensions must be contiguous, got strides {strides}"
    )

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    # mock kv cache contains just the pages needed by this prefill
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)
    # we simply sequentially index the pages needed by this prefill
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device=block_tables_prefill.device,
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        strides[0],
        strides[1],
        strides[2],
        k_cache_stride,
        kv_cache_stride,
        head_stride,
        num_kv_heads,
    )
    return mock_kv_cache, mock_block_table


class BatchDCPPrefillWrapper:
    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        dcp_a2a: bool = False,
    ):
        if dcp_a2a:
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)
        self._context = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, get_kv_cache_layout()
        )
        self._new_tokens = BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer)

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
    ):
        """Plan the prefill operation with given parameters."""
        self._context.plan(
            qo_indptr=qo_indptr_cpu,
            paged_kv_indptr=paged_kv_indptr_cpu,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len_cpu,
            num_qo_heads=num_qo_heads * dcp_world_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=False,  # This is context run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_cache_dtype,
            fixed_split_size=prefill_fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self._new_tokens.plan(
            qo_indptr=qo_indptr_cpu,
            kv_indptr=qo_indptr_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            causal=True,  # This is newtokens run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
        )

    def run(
        self,
        layer: torch.nn.Module,
        prefill_query: torch.Tensor,
        kv_cache_permute: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
    ):
        prefill_query_across_dcp = get_dcp_group().all_gather(
            prefill_query.contiguous(), dim=1
        )
        output_context_tmp, lse_context_tmp = self._context.run(
            prefill_query_across_dcp,
            kv_cache_permute,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            return_lse=True,
        )
        output_context, lse_context = self._dcp_combine(
            output_context_tmp,
            lse_context_tmp,
            get_dcp_group(),
            return_lse=True,
        )
        lse_context = lse_context.transpose(0, 1).contiguous()

        output_query, lse_query = self._new_tokens.run(
            prefill_query,
            key,
            value,
            return_lse=True,
        )
        lse_query = lse_query.transpose(0, 1).contiguous()

        merge_attn_states(
            out,
            output_context,
            lse_context,
            output_query,
            lse_query,
        )
        return out


class FlashInferBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Note: Not sure for all platforms, but on Blackwell,
        # only support a page size of 16, 32, 64.
        return [16, 32, 64]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @staticmethod
    def get_impl_cls() -> type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMetadataBuilder"]:
        return FlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "nvfp4":
            # Packed layout: fp4 data + fp8 block scales in last dim
            last_dim = nvfp4_kv_cache_full_dim(head_size)
            return (num_blocks, 2, block_size, num_kv_heads, last_dim)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, 2, num_kv_heads, num_layers, block_size, head_size)
            return (1, 2, 4, 0, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        elif kv_cache_dtype == "nvfp4":
            return torch.uint8
        else:
            raise ValueError(f"Unrecognized dtype: {kv_cache_dtype}")

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256, 512]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(7, 5) and capability <= DeviceCapability(
            12, 1
        )

    @classmethod
    def supports_sink(cls) -> bool:
        """FlashInfer supports sinks when TRTLLM attention is available (SM100)."""
        from vllm.utils.flashinfer import (
            force_use_trtllm_attention,
            supports_trtllm_attention,
        )

        # Respect explicit disable flag (e.g.,
        # --attention-config.use_trtllm_attention=0)
        if force_use_trtllm_attention() is False:
            return False

        # Check if TRTLLM is supported on this platform
        return supports_trtllm_attention()

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 10:
            return "HND"
        return None

    forward_includes_kv_cache_update: bool = False


@dataclass
class FIPrefill:
    """Metadata for the native FlashInfer prefill pathway (non-TRTLLM)."""

    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper


@dataclass
class FIDecode:
    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""

    wrapper: BatchDecodeWithPagedKVCacheWrapper


@dataclass
class TRTLLMPrefill:
    """Metadata for the TRTLLM prefill pathway."""

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to prefill requests.
    Shape: [num_prefills, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to prefill requests.
    Shape: [num_prefills]
    """

    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor

    max_q_len: int
    """
    The maximum query length *among prefill requests*.
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


@dataclass
class TRTLLMDecode:
    """Metadata for the TRTLLM decode pathway."""

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to decode requests.
    Shape: [num_decodes, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to decode requests.
    Shape: [num_decodes]
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


@dataclass
class FlashInferMetadata:
    num_actual_tokens: int
    """Total number of tokens in the batch (excluding padding)."""

    slot_mapping: torch.Tensor
    """Tensor for writing K/V to the cache. Shape: [num_actual_tokens]"""

    q_data_type: torch.dtype

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    prefill: FIPrefill | TRTLLMPrefill | None
    """
    Holds the metadata for the prefill portion of the batch.
    Will be `None` if `num_prefill_tokens == 0`.
    """

    decode: FIDecode | TRTLLMDecode | None
    """
    Holds the metadata for the decode portion of the batch.
    Will be `None` if `num_decode_tokens == 0`.
    """

    # --- Special Case: Cascade Attention ---

    use_cascade: bool
    """
    If True, the entire batch is a cascade attention call, and the
    `prefill` and `decode` fields will both be None.
    """

    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.attention_config = vllm_config.attention_config
        self._workspace_buffer = None
        self._prefill_wrapper: (
            BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper | None
        ) = None  # Wrapper for prefill/append
        self._decode_wrapper = None  # Wrapper for decode (general shape)

        if envs.VLLM_BATCH_INVARIANT:
            self.decode_fixed_split_size = 2048
            self.prefill_fixed_split_size = 4096
            self.disable_split_kv = True
        else:
            self.decode_fixed_split_size = -1
            self.prefill_fixed_split_size = -1
            self.disable_split_kv = False

        self.compilation_config = vllm_config.compilation_config
        max_num_pages_per_req = cdiv(
            self.model_config.max_model_len, self.kv_cache_spec.block_size
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req
        speculative_config = vllm_config.speculative_config
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        self.enable_cuda_graph = (
            self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        )
        if self.enable_cuda_graph:
            # For full cudagraph capture, one `decode_wrapper` for each batch
            # size is needed for FlashInfer.
            self._decode_wrappers_cudagraph: dict[
                int, BatchDecodeWithPagedKVCacheWrapper
            ] = {}
            self._decode_cudagraph_max_bs = (1 + num_spec_tokens) * max_num_reqs
            if self.compilation_config.max_cudagraph_capture_size is not None:
                self._decode_cudagraph_max_bs = min(
                    self._decode_cudagraph_max_bs,
                    self.compilation_config.max_cudagraph_capture_size,
                )
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.dcp_kv_cache_interleave_size = (
                vllm_config.parallel_config.dcp_kv_cache_interleave_size
            )
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.dcp_kv_cache_interleave_size = 1
        self.use_dcp = self.dcp_world_size > 1
        self.dcp_a2a = (
            self.use_dcp and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )

        self.num_qo_heads = self.model_config.get_num_attention_heads(
            self.vllm_config.parallel_config
        )

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        self.page_size = self.kv_cache_spec.block_size

        if self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE:
            # Prefer the dtype string the layer group resolved to: with
            # per-layer mixed KV dtypes (kv_cache_dtype_skip_layers
            # overrides) the global cache_config.cache_dtype no longer
            # describes every group. Cannot use self.kv_cache_spec.dtype
            # because kv_cache_spec storage dtype may not be the same as
            # the op dtype (uint8 vs fp8_e4m3).
            self.cache_dtype = (
                getattr(self.kv_cache_spec, "cache_dtype_str", None)
                or self.cache_config.cache_dtype
            )
            self.is_kvcache_nvfp4 = self.cache_dtype == "nvfp4"
            self.use_fa2_nvfp4_kv = False
            if self.is_kvcache_nvfp4:
                self.use_fa2_nvfp4_kv = current_platform.is_device_capability_family(
                    120
                )
                if self.use_fa2_nvfp4_kv:
                    self.kv_cache_dtype = FlashInferBackend.get_dtype_for_flashinfer(
                        "nvfp4"
                    )
                # trtllm-gen FP4 FMHA kernels only exist for sm100f (sm_100/sm_103).
                # SM12x routes NVFP4 KV through FlashInfer FA2 instead.
                elif current_platform.is_device_capability_family(100):
                    # For NVFP4, kv_cache_dtype stays as the string "nvfp4"
                    # which is passed to FlashInferImpl.
                    self.kv_cache_dtype = self.cache_dtype
                else:
                    raise ValueError(
                        "--kv-cache-dtype nvfp4 requires sm100f or SM12x, "
                        "please try a different dtype or remove it."
                    )
            else:
                self.kv_cache_dtype = FlashInferBackend.get_dtype_for_flashinfer(
                    self.cache_dtype
                )
        else:
            self.cache_dtype = "auto"
            self.is_kvcache_nvfp4 = False
            self.use_fa2_nvfp4_kv = False
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        # Use model dtype as q dtype when TRTLLM attn is not supported, or
        # --attention-config.disable_flashinfer_q_quantization is set to 1. Otherwise,
        # try to use fp8 q if kv cache is fp8, and will fall back to model dtype
        # if TRTLLM attention kernel is not used when building attn metadata
        can_use_trtllm = can_use_trtllm_attention(self.num_qo_heads, self.num_kv_heads)

        if (
            can_use_trtllm
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
            and not self.use_fa2_nvfp4_kv
        ):
            if self.is_kvcache_nvfp4:
                # NVFP4 KV cache uses FP8 quantized queries
                self.q_data_type = FlashInferBackend.get_dtype_for_flashinfer(
                    "fp8_e4m3"
                )
            else:
                self.q_data_type = self.kv_cache_dtype
        else:
            self.q_data_type = self.model_config.dtype

        # Prefer TRTLLM attention for decoding in all cases.
        # This allows us to use AttentionCGSupport.UNIFORM_BATCH mode.
        self.use_trtllm_decode_attention = can_use_trtllm and not (
            self.use_fa2_nvfp4_kv
        )
        self._init_reorder_batch_threshold(
            1, supports_spec_as_decode=self.use_trtllm_decode_attention
        )

        if self.use_fa2_nvfp4_kv and self.use_dcp:
            raise NotImplementedError(
                "FlashInfer FA2 NVFP4 KV on SM12x is not wired for DCP yet."
            )
        if self.use_fa2_nvfp4_kv:
            _ensure_vllm_nvfp4_kv_deswizzle_flag()
            logger.info_once(
                "Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x "
                "with vLLM V-scale-factor deswizzle enabled."
            )

        self.vo_split = _vo_split_factor(
            self.head_dim, self.use_fa2_nvfp4_kv
        )
        if self.vo_split > 1:
            # BatchDecodeWithPagedKVCacheWrapper.plan() has no head_dim_vo,
            # so route every request through the VO-split-planned prefill
            # wrapper: threshold 0 classifies nothing as decode, and a
            # causal qo_len==1 prefill computes exactly what decode would.
            self.reorder_batch_threshold = 0
            logger.info_once(
                "FA2 VO split (%s KV): head_size %d runs as %d passes of "
                "head_dim_vo=%d; decode requests use the prefill wrapper.",
                self.cache_dtype,
                self.head_dim,
                self.vo_split,
                self.head_dim // self.vo_split,
            )

        self._cascade_wrapper = None  # Wrapper for cascade attention

        # Global hyperparameters shared by all attention layers
        # TODO: discard this for trtllm-gen backend
        self.global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, FlashInferImpl)
        )
        self.sm_scale = self.global_hyperparameters.sm_scale
        self.window_left = self.global_hyperparameters.window_left
        self.logits_soft_cap = self.global_hyperparameters.logits_soft_cap
        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks and not can_use_trtllm:
            raise NotImplementedError(
                "FlashInfer backend currently does not support attention "
                "sinks, please use trtllm on blackwell or flash attention on "
                "earlier GPUs."
            )
        # Preparing persistent buffers
        # Since we do not have explicit synchronization in ModelRunnerV2, we do not pin
        # reused CPU buffers to avoid a race condition between step N async copies to
        # GPU and step N+1 buffer updates.
        self.pin_memory = (
            not vllm_config.use_v2_model_runner and is_pin_memory_available()
        )
        self.paged_kv_indptr = self._make_buffer(max_num_reqs + 1)
        self.paged_kv_indptr_cpu_buffer = torch.zeros_like(
            self.paged_kv_indptr.cpu, pin_memory=self.pin_memory
        )  # Extra buffer for mutable paged_kv_indptr.cpu in cuda graph mode
        self.paged_kv_indices = self._make_buffer(max_num_pages)
        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype = torch.int32
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=True,
        )

    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type["FlashInferMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        """Get the cudagraph support level for FlashInfer attention.

        This depends on whether we can use TRTLLM attention for decodes, since we can
        only do UNIFORM_SINGLE_TOKEN_DECODE if it is unavailable.
        To check this, we must call can_use_trtllm_attention with the number of KV
        heads from the kv_cache_spec. We check all available KV cache specs and
        only return UNIFORM_BATCH if all of them support TRTLLM attention.
        """
        # For UniformTypeKVCacheSpecs, check all contained specs
        kv_specs = (
            kv_cache_spec.kv_cache_specs.values()
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)
            else [kv_cache_spec]
        )
        num_qo_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        has_trtllm_support: bool = len(kv_specs) > 0
        for spec in kv_specs:
            if not isinstance(spec, AttentionSpec):
                # FlashInfer only applies to attention, so we don't consider other types
                # of KV spec (e.g. Mamba) here. This is mostly for type checking.
                continue
            if spec.head_size > 256 and (
                _vllm_flashinfer_vosplit_requested()
                or (
                    spec.kv_quant_mode != KVQuantMode.NONE
                    and current_platform.is_device_capability_family(120)
                    and _vllm_nvfp4_kv_vosplit_requested()
                )
            ):
                # The VO-split group routes decodes through the
                # dynamically planned prefill wrapper, which cudagraph
                # capture cannot replay. Piecewise graphs are unaffected.
                return AttentionCGSupport.NEVER
            if not can_use_trtllm_attention(
                num_qo_heads=num_qo_heads,
                num_kv_heads=spec.num_kv_heads,
            ):
                has_trtllm_support = False
                break

        if has_trtllm_support:
            return AttentionCGSupport.UNIFORM_BATCH
        else:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            buffer_size = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
            if envs.VLLM_BATCH_INVARIANT:
                buffer_size = FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
            self._workspace_buffer = torch.zeros(
                buffer_size, dtype=torch.uint8, device=self.device
            )
        return self._workspace_buffer

    def set_workspace_buffer(self, workspace_buffer: torch.Tensor):
        self._workspace_buffer = workspace_buffer

    def _get_prefill_wrapper(
        self,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        if self._prefill_wrapper is None:
            if self.use_dcp:
                self._prefill_wrapper = BatchDCPPrefillWrapper(
                    workspace_buffer=self._get_workspace_buffer(),
                    dcp_a2a=self.dcp_a2a,
                )
            else:
                if self.use_fa2_nvfp4_kv:
                    backend = "fa2"
                    o_dtype = self.model_config.dtype
                    jit_args, jit_kwargs = _fa2_nvfp4_prefill_jit_args(
                        q_data_type=self.q_data_type,
                        kv_data_type=self.kv_cache_dtype,
                        o_data_type=o_dtype,
                        idtype=torch.int32,
                        head_dim_qk=self.head_dim,
                        # Ctor jit_args override plan-time head_dim_vo, so
                        # the VO split MUST be reflected here too: the 31B
                        # full-NVFP4 smoke crashed at prefill.cuh:3215 with
                        # NUM_MMA_D_VO=32 (vo=512 reached the kernel)
                        # because this arg pinned the symmetric pair while
                        # plan() asked for (512, 256).
                        head_dim_vo=self.head_dim // self.vo_split,
                        use_sliding_window=self.window_left >= 0,
                        use_logits_soft_cap=(self.logits_soft_cap or 0.0) > 0,
                    )
                elif self.is_kvcache_nvfp4:
                    backend = "trtllm-gen"
                    jit_args = None
                    jit_kwargs = None
                else:
                    backend = "auto"
                    jit_args = None
                    jit_kwargs = None
                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(),
                    get_kv_cache_layout(),
                    backend=backend,
                    jit_args=jit_args,
                    jit_kwargs=jit_kwargs,
                )
        assert self._prefill_wrapper is not None
        return self._prefill_wrapper

    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices.gpu
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None
            if self.use_fa2_nvfp4_kv:
                backend = "fa2"
            elif self.is_kvcache_nvfp4:
                backend = "trtllm-gen"
            else:
                backend = "auto"
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                get_kv_cache_layout(),
                use_cuda_graph=use_cudagraph,
                paged_kv_indptr_buffer=paged_kv_indptr,
                paged_kv_indices_buffer=paged_kv_indices,
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,
                # Tensor cores are enabled by default because the perf would be
                # at least as good as cuda cores for all attention ops in latest
                # gpus.
                use_tensor_cores=True,
                backend=backend,
            )

            # save the decode wrapper
            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper
            else:
                self._decode_wrapper = decode_wrapper

        return decode_wrapper

    def _get_cascade_wrapper(self):
        if self._cascade_wrapper is None:
            self._cascade_wrapper = MultiLevelCascadeAttentionWrapper(
                2, self._get_workspace_buffer(), get_kv_cache_layout()
            )
        return self._cascade_wrapper

    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
        page_size: int,
    ) -> torch.Tensor:
        """
        Compute paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len for FlashInfer
        attention.

        Results are stored in self.paged_kv_indptr,
        self.paged_kv_indices, self.paged_kv_last_page_len buffers.

        Returns paged_kv_indices, a GPU tensor with shape [num_actual_pages].
        """
        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr.np[1 : num_reqs + 1],
        )
        # NOTE(woosuk): Because self.paged_kv_indptr_cpu can be modified
        # after this line (e.g., for cuda graphs), we need to copy the data to
        # self.paged_kv_indptr_buffer to avoid race condition.
        self.paged_kv_indptr_cpu_buffer[: num_reqs + 1] = self.paged_kv_indptr.cpu[
            : num_reqs + 1
        ]
        paged_kv_indptr = self.paged_kv_indptr.gpu[: num_reqs + 1]
        paged_kv_indptr.copy_(
            self.paged_kv_indptr_cpu_buffer[: num_reqs + 1], non_blocking=True
        )

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr.np[num_reqs]
        paged_kv_indices = self.paged_kv_indices.gpu[:num_actual_pages]
        _copy_page_indices_kernel[(num_reqs,)](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
            page_size,
            paged_kv_last_page_len_np,
        )
        self.paged_kv_last_page_len.gpu[:num_reqs].copy_(
            self.paged_kv_last_page_len.cpu[:num_reqs], non_blocking=True
        )
        return paged_kv_indices

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        page_size = self.page_size
        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr = common_attn_metadata.query_start_loc
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu

        # Step 1: Decide which dispatch modes to use:
        # - Cascade attention (distinct mode)
        # - Prefill (FI native or TRTLLM)
        # - Decode (FI native or TRTLLM)
        use_cascade = common_prefix_len > 0
        uses_spec_reorder = self.reorder_batch_threshold > 1
        if self.use_fa2_nvfp4_kv:
            prefill_use_trtllm = False
        else:
            prefill_use_trtllm = use_trtllm_attention(
                self.num_qo_heads,
                self.num_kv_heads,
                num_prefill_tokens,
                max_seq_len,
                self.dcp_world_size,
                self.cache_dtype,
                self.q_data_type,
                is_prefill=True,
                force_use_trtllm=self.attention_config.use_trtllm_attention,
                has_sinks=self.has_sinks,
                has_spec=uses_spec_reorder,
            )
        decode_use_trtllm = (
            self.use_trtllm_decode_attention and self.dcp_world_size <= 1
        )

        all_uses_trtllm = (num_prefills == 0 or prefill_use_trtllm) and (
            num_decodes == 0 or decode_use_trtllm
        )

        if not all_uses_trtllm:
            if self.has_sinks:
                raise NotImplementedError(
                    "FlashInfer backend currently does not support attention "
                    "sinks, please use trtllm on blackwell or flash attention "
                    "on earlier GPUs."
                )

            if not self.global_hyperparameters.has_same_window_lefts:
                raise ValueError(
                    "Window left is not the same for all layers. "
                    "One potential fix is to set disable_sliding_window=True"
                )

            assert self.global_hyperparameters.has_same_all_params, (
                "FlashInfer backend currently only supports models in which "
                "all layers share the same values for the following "
                "hyperparameters: `window_left`, `logits_soft_cap`, "
                "`sm_scale`."
            )

            # The q quantization is not supported for non-trtllm attention,
            # fall back to model dtype.
            self.q_data_type = self.model_config.dtype

        # Step 2: Initialize the output metadata
        # Leave prefill/decode/cascade_wrapper empty, to be populated
        # case by case depending on the batch contents and backend selection.
        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            q_data_type=self.q_data_type,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
            prefill=None,
            decode=None,
            cascade_wrapper=None,
        )

        # Guard access to seq_lens_cpu, which may not always be needed
        # and can be expensive to retrieve in async mode.
        # When all attention (both prefill and decode) uses TRTLLM,
        # seq_lens_cpu is not needed since TRTLLM paths use GPU tensors
        # (block_tables, seq_lens) directly.
        needs_seq_lens_cpu = self.use_dcp or use_cascade or not all_uses_trtllm
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None
        seq_lens_np = seq_lens_cpu.numpy() if seq_lens_cpu is not None else None
        num_blocks_np = (
            (seq_lens_np + (page_size - 1)) // page_size
            if seq_lens_np is not None
            else None
        )

        # Adjust seq_lens_cpu for DCP
        if self.use_dcp:
            assert seq_lens_cpu is not None
            if num_prefills > 0:
                qo_indptr_prefill_cpu = (
                    qo_indptr_cpu[num_decodes:] - qo_indptr_cpu[num_decodes]
                )
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                seq_lens_cpu[num_decodes:] = (
                    seq_lens_cpu[num_decodes:] - query_lens_prefill_cpu
                )

            seq_lens_cpu = get_dcp_local_seq_lens(
                seq_lens_cpu,
                self.dcp_world_size,
                self.dcp_rank,
                self.dcp_kv_cache_interleave_size,
            )

        # Adjust num_block_np for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            assert common_prefix_len % page_size == 0
            num_common_kv_blocks = common_prefix_len // page_size
            num_blocks_np -= num_common_kv_blocks

        # Compute paged_kv_indices if necessary
        # paged_kv_indices is only needed for FlashInfer native paths;
        # TRTLLM paths use block_tables directly on GPU.
        needs_paged_kv_indices = use_cascade or not all_uses_trtllm
        if needs_paged_kv_indices:
            assert num_blocks_np is not None
            assert seq_lens_np is not None
            paged_kv_indices = self._compute_flashinfer_kv_metadata(
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                num_reqs,
                page_size,
            )
        else:
            paged_kv_indices = None

        if _spark_kv_trace_should_emit(
            "fi_metadata", layer_names=self.layer_names
        ):
            limit = _spark_kv_trace_int("VLLM_SPARK_KV_TRACE_LIMIT", 4)
            _spark_kv_trace(
                "fi_metadata",
                {
                    "layer_names": self.layer_names,
                    "cache_dtype": str(self.cache_dtype),
                    "kv_cache_dtype": str(self.kv_cache_dtype),
                    "page_size": int(page_size),
                    "head_dim": int(self.head_dim),
                    "num_q_heads": int(self.num_qo_heads),
                    "num_kv_heads": int(self.num_kv_heads),
                    "window_left": int(self.window_left),
                    "num_reqs": int(num_reqs),
                    "num_actual_tokens": int(num_actual_tokens),
                    "num_decodes": int(num_decodes),
                    "num_decode_tokens": int(num_decode_tokens),
                    "num_prefills": int(num_prefills),
                    "num_prefill_tokens": int(num_prefill_tokens),
                    "slot_mapping_head": _spark_kv_trace_tensor_head(
                        common_attn_metadata.slot_mapping, limit
                    ),
                    "block_table_head": _spark_kv_trace_tensor_head(
                        block_table_tensor, limit
                    ),
                    "paged_kv_indptr_head": _spark_kv_trace_tensor_head(
                        self.paged_kv_indptr.cpu[: num_reqs + 1], limit
                    ),
                    "paged_kv_indices_head": _spark_kv_trace_tensor_head(
                        paged_kv_indices, limit
                    ),
                    "paged_kv_last_page_len_head": _spark_kv_trace_tensor_head(
                        self.paged_kv_last_page_len.cpu[:num_reqs], limit
                    ),
                    "prefill_use_trtllm": bool(prefill_use_trtllm),
                    "decode_use_trtllm": bool(decode_use_trtllm),
                },
            )

        # Early-out for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            # Grab the blocks of the shared prefix from the first request.
            num_common_kv_blocks = common_prefix_len // page_size

            # Create CPU versions directly for cascade (no GPU versions needed)
            shared_qo_indptr_cpu = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indptr_cpu = torch.tensor(
                [0, num_common_kv_blocks], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indices_cpu = block_table_tensor[0, :num_common_kv_blocks]
            shared_kv_last_page_len_cpu = torch.tensor(
                [page_size], dtype=torch.int32, device="cpu"
            )

            # Remove the blocks of the shared prefix from all requests.
            block_table_tensor = block_table_tensor[:, num_common_kv_blocks:]
            num_blocks_np -= num_common_kv_blocks

            assert paged_kv_indices is not None
            paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[: 1 + num_reqs]
            paged_kv_last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_reqs]

            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()
            attn_metadata.cascade_wrapper.plan(
                qo_indptr_arr=[shared_qo_indptr_cpu, qo_indptr_cpu],
                paged_kv_indptr_arr=[shared_kv_page_indptr_cpu, paged_kv_indptr_cpu],
                paged_kv_indices_arr=[shared_kv_page_indices_cpu, paged_kv_indices],
                paged_kv_last_page_len=[
                    shared_kv_last_page_len_cpu,
                    paged_kv_last_page_len_cpu,
                ],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                causal=True,
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_cache_dtype,
            )
            return attn_metadata

        # Step 3: Handle prefill and decode pathways case by case
        ## PREFILL PATHWAY
        if num_prefills > 0:
            # Slices for shared prefill metadata
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            )
            assert qo_indptr_prefill_cpu.shape[0] == num_prefills + 1

            if prefill_use_trtllm:
                # Create GPU versions
                qo_indptr_prefill_gpu = (
                    qo_indptr[prefill_start:] - qo_indptr[prefill_start]
                )
                # Compute cum_seq_lens_kv on GPU to avoid CPU sync.
                # This is the cumulative sum of the number of KV cache
                # blocks per prefill request.
                prefill_seq_lens = seq_lens[prefill_start:]
                num_blocks_per_req = (prefill_seq_lens + page_size - 1) // page_size
                paged_kv_indptr_prefill_gpu = self.paged_kv_indptr.gpu[
                    prefill_start : num_reqs + 1
                ]
                # Assign to slice to avoid cpu sync.
                paged_kv_indptr_prefill_gpu[:1] = 0
                torch.cumsum(
                    num_blocks_per_req,
                    dim=0,
                    out=paged_kv_indptr_prefill_gpu[1:],
                )
                # Compute max_q_len for prefill requests
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                max_q_len_prefill = int(query_lens_prefill_cpu.max().item())
                attn_metadata.prefill = TRTLLMPrefill(
                    block_tables=block_table_tensor[prefill_start:],
                    seq_lens=seq_lens[prefill_start:],
                    cum_seq_lens_q=qo_indptr_prefill_gpu,
                    cum_seq_lens_kv=paged_kv_indptr_prefill_gpu,
                    max_q_len=max_q_len_prefill,
                    max_seq_len=max_seq_len,
                )
            else:
                prefill_wrapper = self._get_prefill_wrapper()
                # Slicing CPU buffers that are only needed for FI native prefills
                paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[
                    prefill_start:num_reqs
                ]
                assert paged_kv_last_page_len_prefill_cpu.shape[0] == num_prefills
                paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[
                    prefill_start : num_reqs + 1
                ]
                assert paged_kv_indptr_prefill_cpu.shape[0] == num_prefills + 1
                if self.use_dcp:
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    prefill_wrapper.plan(
                        qo_indptr_cpu=qo_indptr_prefill_cpu,
                        paged_kv_indptr_cpu=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len_cpu=paged_kv_last_page_len_prefill_cpu,
                        page_size=self.page_size,
                        num_qo_heads=self.num_qo_heads,
                        dcp_world_size=self.dcp_world_size,
                        num_kv_heads=self.num_kv_heads,
                        head_dim=self.head_dim,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type,
                        kv_cache_dtype=self.kv_cache_dtype,
                        prefill_fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                else:
                    assert isinstance(
                        prefill_wrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                    )
                    # The SM100 trtllm NVFP4 path only supports FP8 output.
                    # The SM12x FA2 path writes model dtype directly.
                    o_dtype = (
                        FP8_DTYPE
                        if self.is_kvcache_nvfp4 and not self.use_fa2_nvfp4_kv
                        else self.model_config.dtype
                    )
                    prefill_wrapper.plan(
                        qo_indptr=qo_indptr_prefill_cpu,
                        paged_kv_indptr=paged_kv_indptr_prefill_cpu,
                        paged_kv_indices=paged_kv_indices,
                        paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu,
                        num_qo_heads=self.num_qo_heads,
                        num_kv_heads=self.num_kv_heads,
                        head_dim_qk=self.head_dim,
                        # == head_dim_qk unless the NVFP4 VO split is active;
                        # then the impl runs this wrapper once per V half.
                        head_dim_vo=self.head_dim // self.vo_split,
                        page_size=self.page_size,
                        causal=True,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type,
                        kv_data_type=self.kv_cache_dtype,
                        o_data_type=o_dtype,
                        fixed_split_size=self.prefill_fixed_split_size,
                        disable_split_kv=self.disable_split_kv,
                    )
                    prefill_wrapper.vllm_prefill_fixed_split_size = (
                        self.prefill_fixed_split_size
                    )
                    prefill_wrapper.vllm_disable_split_kv = self.disable_split_kv
                attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)

        ## DECODE PATHWAY
        if num_decodes > 0:
            assert self.vo_split == 1, (
                "NVFP4 VO split routes decodes through the prefill wrapper "
                "(reorder_batch_threshold=0); the decode pathway should be "
                "unreachable."
            )
            if decode_use_trtllm:
                assert num_decode_tokens % num_decodes == 0, (
                    "TRTLLM decode requires uniform query lengths per request. "
                    f"Got {num_decode_tokens=} and {num_decodes=}."
                )
                attn_metadata.decode = TRTLLMDecode(
                    block_tables=block_table_tensor[:num_decodes],
                    seq_lens=seq_lens[:num_decodes],
                    max_seq_len=max_seq_len,
                )
            else:
                assert seq_lens_cpu is not None
                pure_decode = num_prefills == 0
                use_cudagraph = (
                    self.enable_cuda_graph
                    and pure_decode
                    and num_decode_tokens <= self._decode_cudagraph_max_bs
                )
                num_input_tokens = num_decode_tokens

                decode_wrapper = self._get_decode_wrapper(
                    num_input_tokens, use_cudagraph
                )
                # Use the persistent buffer with padding length,
                # instead of the same address but chunked version
                # in atten_metadata when using cudagraph.
                # The SM100 trtllm NVFP4 path only supports FP8 output.
                # The SM12x FA2 path writes model dtype directly.
                o_dtype = (
                    FP8_DTYPE
                    if self.is_kvcache_nvfp4 and not self.use_fa2_nvfp4_kv
                    else self.model_config.dtype
                )
                fast_plan_decode(
                    decode_wrapper,
                    indptr_cpu=self.paged_kv_indptr.cpu[: num_input_tokens + 1],
                    indices=paged_kv_indices,
                    last_page_len_cpu=self.paged_kv_last_page_len.cpu[
                        :num_input_tokens
                    ],
                    num_qo_heads=self.num_qo_heads * self.dcp_world_size,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    # Disable flashinfer's pos encoding and use vllm's rope.
                    pos_encoding_mode="NONE",
                    sm_scale=self.sm_scale,
                    window_left=self.window_left,
                    logits_soft_cap=self.logits_soft_cap,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.kv_cache_dtype,
                    o_data_type=o_dtype,
                    fixed_split_size=self.decode_fixed_split_size,
                    disable_split_kv=self.disable_split_kv,
                )
                attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype:
            # TODO: The cascade wrapper currently does not support setting
            # kv cache dtype to something different from query dtype.
            return False
        # TODO: Cascade attention doesn't work, disable it for now
        # return use_cascade_attention(*args, **kwargs)
        return False


class FlashInferImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.window_left = (
            self.sliding_window[0] if self.sliding_window is not None else -1
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.is_kvcache_nvfp4 = kv_cache_dtype == "nvfp4"
        self.use_fa2_nvfp4_kv = (
            self.is_kvcache_nvfp4
            and current_platform.is_device_capability_family(120)
        )
        if self.use_fa2_nvfp4_kv:
            _ensure_vllm_nvfp4_kv_deswizzle_flag()
        self.fp4_data_dim = head_size // 2 if self.is_kvcache_nvfp4 else 0
        self.vo_split = _vo_split_factor(
            head_size, self.use_fa2_nvfp4_kv
        )
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferImpl"
            )

        self.sinks: torch.Tensor | None = None
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, but got "
                    f"{sinks.shape[0]}."
                )
            self.sinks = sinks

        self.support_trtllm_attn = can_use_trtllm_attention(num_heads, num_kv_heads)
        vllm_config = get_current_vllm_config_or_none()
        self.supports_quant_query_input = (
            self.support_trtllm_attn
            and vllm_config is not None
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        )
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        self.o_sf_scale: float | None = None

        # Pre-allocated FP8 output buffer for SM100 TRTLLM NVFP4 without
        # fused output quant. The SM12x FA2 path writes model dtype directly.
        if (
            self.is_kvcache_nvfp4
            and not self.use_fa2_nvfp4_kv
            and vllm_config is not None
        ):
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self._nvfp4_fp8_out = torch.empty(
                (max_num_tokens, num_heads, head_size),
                dtype=FP8_DTYPE,
                device="cuda",
            )
        else:
            self._nvfp4_fp8_out = None

        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        if dcp_a2a:
            self.dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self.dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return (
            self.support_trtllm_attn
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and quant_key in (kFp8StaticTensorSym, kNvfp4Dynamic)
        )

    # FlashInfer requires attention sinks to be float32
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self.sinks is not None and self.sinks.dtype != torch.float32:
            self.sinks = self.sinks.to(torch.float32)

    def _run_vo_split_prefill(
        self,
        wrapper: BatchPrefillWithPagedKVCacheWrapper,
        query: torch.Tensor,
        kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_sf: tuple[torch.Tensor, torch.Tensor] | None,
        out: torch.Tensor,
        *,
        k_scale: float,
        v_scale: float,
    ) -> None:
        """Multi-pass FA2 run for head_size > 256 (any KV dtype).

        The wrapper is planned with head_dim_vo = head_size // split, and
        each pass consumes a zero-copy head-dim slice of V (and, for
        NVFP4, of the V scale factors): S = Q @ K^T and the softmax are
        recomputed identically per pass, so the per-pass outputs
        concatenate exactly (no LSE merge). narrow() keeps the full
        tensor's strides, which the FA2 path requires: the K/V
        stride-equality check passes and the run sizes its output from
        the V view's width. NVFP4 additionally requires the linear V-SF
        layout (enforced in _vo_split_factor) — the swizzled layout does
        not slice along the head dim.
        """
        split = self.vo_split
        head_chunk = self.head_size // split
        if self.is_kvcache_nvfp4:
            assert isinstance(kv_cache, tuple) and kv_sf is not None
            k_cache, v_cache = kv_cache
            k_sf, v_sf = kv_sf
            data_step = head_chunk // 2  # packed e2m1, 2 elements per byte
            sf_step = head_chunk // 16  # one fp8 scale per 16 elements
        else:
            # Dense bf16/fp16 or fp8 KV: a stacked (num_pages, 2, ...)
            # cache or an explicit (k, v) pair; V slices are plain
            # element-width views and there are no scale factors.
            if isinstance(kv_cache, tuple):
                k_cache, v_cache = kv_cache
            else:
                k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
            k_sf = v_sf = None
            data_step = head_chunk
            sf_step = 0
        for i in range(split):
            v_cache_i = v_cache.narrow(-1, i * data_step, data_step)
            kv_sf_i = (
                (k_sf, v_sf.narrow(-1, i * sf_step, sf_step))
                if v_sf is not None
                else None
            )
            # The kernel needs a contiguous output; write into a chunk
            # buffer and copy into the strided slice of the full output.
            out_i = torch.empty(
                (*out.shape[:-1], head_chunk),
                dtype=out.dtype,
                device=out.device,
            )
            wrapper.run(
                query,
                (k_cache, v_cache_i),
                k_scale=k_scale,
                v_scale=v_scale,
                out=out_i,
                kv_cache_sf=kv_sf_i,
            )
            out.narrow(-1, i * head_chunk, head_chunk).copy_(out_i)

    def _spark_nvfp4_prefill_fresh_wrapper_replay(
        self,
        *,
        layer: torch.nn.Module,
        layer_name: str | None,
        live_wrapper: BatchPrefillWithPagedKVCacheWrapper,
        prefill_query: torch.Tensor,
        kv_cache_permute: tuple[torch.Tensor, ...],
        kv_cache_sf: tuple[torch.Tensor, ...] | None,
        live_out: torch.Tensor,
    ) -> None:
        if (
            not self.is_kvcache_nvfp4
            or not self.use_fa2_nvfp4_kv
            or not _spark_nvfp4_prefill_fresh_wrapper_replay_enabled()
        ):
            return
        if not spark_tensor_trace_should_emit(
            "flashinfer_wrapper_prefill_fresh_replay", layer_name
        ):
            return

        payload: dict[str, object] = {
            "layer_name": layer_name,
            "kv_cache_dtype": str(self.kv_cache_dtype),
            "window_left": int(self.window_left),
            "head_dim": int(self.head_size),
            "num_q_heads": int(self.num_heads),
            "num_kv_heads": int(self.num_kv_heads),
            "live_wrapper_type": type(live_wrapper).__name__,
            "live_out": spark_trace_last_token_summary(live_out),
        }
        try:
            qo_indptr = getattr(live_wrapper, "_qo_indptr_buf", None)
            paged_kv_indptr = getattr(live_wrapper, "_paged_kv_indptr_buf", None)
            paged_kv_indices = getattr(live_wrapper, "_paged_kv_indices_buf", None)
            paged_kv_last_page_len = getattr(
                live_wrapper, "_paged_kv_last_page_len_buf", None
            )
            if (
                qo_indptr is None
                or paged_kv_indptr is None
                or paged_kv_indices is None
                or paged_kv_last_page_len is None
            ):
                payload["replay_error"] = "missing_wrapper_plan_buffers"
                spark_tensor_trace("flashinfer_wrapper_prefill_fresh_replay", payload)
                return

            batch_size = int(getattr(live_wrapper, "_batch_size", len(qo_indptr) - 1))
            if batch_size <= 0:
                payload["replay_error"] = "empty_batch"
                spark_tensor_trace("flashinfer_wrapper_prefill_fresh_replay", payload)
                return

            num_pages = int(paged_kv_indptr[batch_size].detach().cpu().item())
            qo_indptr = qo_indptr[: batch_size + 1]
            paged_kv_indptr = paged_kv_indptr[: batch_size + 1]
            paged_kv_indices = paged_kv_indices[:num_pages]
            paged_kv_last_page_len = paged_kv_last_page_len[:batch_size]

            workspace_mb = _spark_kv_trace_int(
                "VLLM_SPARK_NVFP4_FRESH_WRAPPER_WORKSPACE_MB", 256
            )
            workspace_bytes = max(1, workspace_mb) * 1024 * 1024
            workspace = torch.empty(
                (workspace_bytes,), dtype=torch.uint8, device=prefill_query.device
            )
            fresh_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                get_kv_cache_layout(),
                backend="fa2",
            )

            live_fixed_split_size = int(
                getattr(live_wrapper, "vllm_prefill_fixed_split_size", -1)
            )
            live_disable_split_kv = bool(
                getattr(live_wrapper, "vllm_disable_split_kv", False)
            )
            replay_q_data_type = getattr(
                live_wrapper, "_cached_q_data_type", prefill_query.dtype
            )
            replay_kv_data_type = getattr(
                live_wrapper, "_cached_kv_data_type", self.kv_cache_dtype
            )
            replay_o_data_type = getattr(
                live_wrapper, "_cached_o_data_type", prefill_query.dtype
            )
            fresh_wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                page_size=kv_cache_permute[0].shape[
                    2 if get_kv_cache_layout() == "HND" else 1
                ],
                causal=True,
                sm_scale=self.scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                q_data_type=replay_q_data_type,
                kv_data_type=replay_kv_data_type,
                o_data_type=replay_o_data_type,
                fixed_split_size=live_fixed_split_size,
                disable_split_kv=live_disable_split_kv,
            )
            fresh_out = torch.empty_like(prefill_query)
            fresh_wrapper.run(
                prefill_query,
                kv_cache_permute,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                out=fresh_out,
                kv_cache_sf=kv_cache_sf,
            )
            payload.update(
                {
                    "batch_size": batch_size,
                    "num_pages": num_pages,
                    "workspace_mb": workspace_mb,
                    "fixed_split_size": live_fixed_split_size,
                    "disable_split_kv": live_disable_split_kv,
                    "q_data_type": str(replay_q_data_type),
                    "kv_data_type": str(replay_kv_data_type),
                    "o_data_type": str(replay_o_data_type),
                    "fresh_out": spark_trace_last_token_summary(fresh_out),
                    "fresh_vs_live": _spark_tensor_trace_compare_payload(
                        fresh_out, live_out
                    ),
                    "fresh_wrapper_backend": getattr(
                        fresh_wrapper, "_backend", "<unknown>"
                    ),
                    "fresh_cached_module": type(
                        getattr(fresh_wrapper, "_cached_module", None)
                    ).__name__,
                    "live_wrapper_backend": getattr(
                        live_wrapper, "_backend", "<unknown>"
                    ),
                    "live_cached_module": type(
                        getattr(live_wrapper, "_cached_module", None)
                    ).__name__,
                    "fresh_wrapper_plan": _spark_trace_prefill_wrapper_payload(
                        fresh_wrapper
                    ),
                    "live_wrapper_plan": _spark_trace_prefill_wrapper_payload(
                        live_wrapper
                    ),
                }
            )
        except Exception as exc:
            payload["replay_error"] = type(exc).__name__
            payload["replay_error_message"] = str(exc)
            logger.exception(
                "Failed Spark fresh FlashInfer prefill replay for %s", layer_name
            )
        spark_tensor_trace("flashinfer_wrapper_prefill_fresh_replay", payload)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: KV cache tensor with different possible shapes:
                - NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
                - HND: [num_blocks, 2, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # Ensure query dtype matches the expected dtype from attention metadata
        assert attn_metadata.q_data_type == query.dtype, (
            f"Query dtype mismatch: expected {attn_metadata.q_data_type}, "
            f"got {query.dtype}"
        )

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._v_scale_float

        prefill_use_trtllm = isinstance(attn_metadata.prefill, TRTLLMPrefill)
        decode_use_trtllm = isinstance(attn_metadata.decode, TRTLLMDecode)

        # The attn+quant fusion happens when output_scale is provided.
        if output_scale is None:
            assert output_block_scale is None, (
                "output_block_scale is not supported when fusion has not happened"
            )
        else:
            assert attn_metadata.q_data_type == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened."
            )
            assert (attn_metadata.num_prefills == 0 or prefill_use_trtllm) and (
                attn_metadata.num_decodes == 0 or decode_use_trtllm
            ), "Must use TRT-LLM attn"

            if output.dtype == FP8_DTYPE:
                assert output_block_scale is None, (
                    "output_block_scale should not be provided for fp8 output"
                )
            elif output.dtype == FP4_DTYPE:
                assert output_block_scale is not None, (
                    "output_block_scale is required for nvfp4 output"
                )
            else:
                raise ValueError(f"Unsupported output dtype: {output.dtype}")

            # TRTLLM attn kernel requires to scale to pass as a host scalar,
            # store the o scale as a host scalar in warmup run with cuda graph
            # not enabled
            if layer._o_scale_float is None:
                layer._o_scale_float = output_scale.cpu().item()
                if output.dtype == FP8_DTYPE:
                    self.bmm2_scale = self.bmm2_scale / layer._o_scale_float
                elif output.dtype == FP4_DTYPE:
                    self.o_sf_scale = layer._o_scale_float

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens
        layer_name = _spark_kv_trace_layer_name(layer)

        # FlashInfer treats uint8 KV cache as NVFP4. vLLM stores FP8 KV cache
        # as uint8 bytes, so pass FP8 caches with their logical dtype.
        if not self.is_kvcache_nvfp4 and kv_cache.dtype == torch.uint8:
            fp8_view_dtype = None
            if self.kv_cache_dtype in ("fp8", "fp8_e4m3", torch.float8_e4m3fn):
                fp8_view_dtype = torch.float8_e4m3fn
            elif self.kv_cache_dtype in ("fp8_e5m2", torch.float8_e5m2):
                fp8_view_dtype = torch.float8_e5m2
            if fp8_view_dtype is not None:
                kv_cache = kv_cache.view(fp8_view_dtype)

        # Inputs and outputs may be padded for CUDA graphs
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        if attn_metadata.use_cascade:
            # Cascade attention (rare case).
            assert attn_metadata.cascade_wrapper is not None
            output.copy_(attn_metadata.cascade_wrapper.run(query, kv_cache))
            return output

        # When using spec decoding, num_decodes can be < num_decode_tokens
        # because some decode requests may have more than one query token.
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)  # HND and contiguous
        # Fix degenerate strides on any size-1 dimension (e.g. num_kv_heads=1
        # with TP=8).  PyTorch permits non-canonical strides on size-1 dims;
        # CUDA TMA requires ≥16-byte alignment on all non-outermost strides.
        # canonicalize_singleton_dim_strides patches metadata via as_strided —
        # zero-copy.  See vllm.utils.torch_utils.
        fixed = canonicalize_singleton_dim_strides(kv_cache_permute)
        if fixed is not kv_cache_permute:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashInfer): "
                "shape=%s, strides before=%s, strides after=%s",
                kv_cache_permute.shape,
                kv_cache_permute.stride(),
                fixed.stride(),
            )
        kv_cache_permute = fixed

        # For NVFP4, the kv_cache last dim is full_dim (data + scale packed).
        # Split into correctly-strided data and scale views.
        nvfp4_kv_data = None
        nvfp4_kv_block_scales = None
        if self.is_kvcache_nvfp4:
            nvfp4_kv_data, nvfp4_kv_block_scales = nvfp4_kv_cache_split_views(
                kv_cache_permute
            )
            if _spark_kv_trace_should_emit(
                "kv_read_views_nvfp4", layer_name=layer_name
            ):
                page_size = int(kv_cache_permute.shape[2])
                _spark_kv_trace(
                    "kv_read_views_nvfp4",
                    {
                        "layer_name": layer_name,
                        "kv_cache_dtype": str(self.kv_cache_dtype),
                        "kv_cache_shape": list(kv_cache.shape),
                        "kv_cache_stride": list(kv_cache.stride()),
                        "kv_cache_permute_shape": list(kv_cache_permute.shape),
                        "kv_cache_permute_stride": list(kv_cache_permute.stride()),
                        "data_views": _spark_kv_trace_views_info(nvfp4_kv_data),
                        "scale_views": _spark_kv_trace_views_info(
                            nvfp4_kv_block_scales
                        ),
                        "slot_mapping_head": _spark_kv_trace_tensor_head(
                            attn_metadata.slot_mapping
                        ),
                        "slot_samples": _spark_kv_trace_slot_samples(
                            nvfp4_kv_data,
                            nvfp4_kv_block_scales,
                            attn_metadata.slot_mapping,
                            page_size,
                        ),
                        "num_actual_tokens": int(num_actual_tokens),
                        "num_decodes": int(attn_metadata.num_decodes),
                        "num_decode_tokens": int(num_decode_tokens),
                        "num_prefills": int(attn_metadata.num_prefills),
                        "num_prefill_tokens": int(num_prefill_tokens),
                        "window_left": int(self.window_left),
                        "head_dim": int(self.head_size),
                        "num_q_heads": int(self.num_heads),
                        "num_kv_heads": int(self.num_kv_heads),
                    },
                )
        if spark_tensor_trace_should_emit("flashinfer_attn_input", layer_name):
            spark_tensor_trace(
                "flashinfer_attn_input",
                {
                    "layer_name": layer_name,
                    "kv_cache_dtype": str(self.kv_cache_dtype),
                    "is_kvcache_nvfp4": bool(self.is_kvcache_nvfp4),
                    "use_fa2_nvfp4_kv": bool(self.use_fa2_nvfp4_kv),
                    "window_left": int(self.window_left),
                    "head_dim": int(self.head_size),
                    "num_q_heads": int(self.num_heads),
                    "num_kv_heads": int(self.num_kv_heads),
                    "num_actual_tokens": int(num_actual_tokens),
                    "num_decodes": int(attn_metadata.num_decodes),
                    "num_decode_tokens": int(num_decode_tokens),
                    "num_prefills": int(attn_metadata.num_prefills),
                    "num_prefill_tokens": int(num_prefill_tokens),
                    "query_last": spark_trace_last_token_summary(query),
                    "key_last": spark_trace_last_token_summary(key),
                    "value_last": spark_trace_last_token_summary(value),
                },
            )

        use_dcp = self.dcp_world_size > 1

        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back.
        if num_prefill_tokens > 0:
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens

            if not prefill_use_trtllm:
                assert isinstance(attn_metadata.prefill, FIPrefill)
                prefill_wrapper = attn_metadata.prefill.wrapper
                assert prefill_wrapper is not None
                if use_dcp:
                    assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                    assert prefill_wrapper._context._window_left == self.window_left
                    assert prefill_wrapper._context._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._context._sm_scale == self.scale
                    assert not prefill_wrapper._context._causal
                    assert prefill_wrapper._new_tokens._window_left == self.window_left
                    assert prefill_wrapper._new_tokens._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._new_tokens._sm_scale == self.scale
                    assert prefill_wrapper._new_tokens._causal

                    prefill_wrapper.run(
                        layer,
                        prefill_query,
                        kv_cache_permute,
                        key[num_decode_tokens:],
                        value[num_decode_tokens:],
                        out=output[num_decode_tokens:],
                    )
                else:
                    assert isinstance(
                        prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper
                    )
                    assert prefill_wrapper._window_left == self.window_left
                    assert prefill_wrapper._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._sm_scale == self.scale
                    assert prefill_wrapper._causal

                    if self.is_kvcache_nvfp4:
                        kv_cache_permute = nvfp4_kv_data
                    kv_cache_sf = (
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    )

                    # SM100 TRTLLM NVFP4 only supports FP8 output. The SM12x
                    # FA2 path writes model dtype directly.
                    needs_fp8_out_prefill = (
                        self.is_kvcache_nvfp4
                        and not self.use_fa2_nvfp4_kv
                        and output.dtype != FP8_DTYPE
                    )
                    if needs_fp8_out_prefill:
                        out_prefill = self._nvfp4_fp8_out[:num_prefill_tokens]
                    else:
                        out_prefill = output[num_decode_tokens:]
                    out_prefill_target = out_prefill
                    uses_contig_out_prefill = (
                        self.is_kvcache_nvfp4
                        and self.use_fa2_nvfp4_kv
                        and _spark_nvfp4_prefill_contig_out_enabled()
                    )
                    if uses_contig_out_prefill:
                        out_prefill = torch.empty_like(prefill_query)
                    dump_out_before = (
                        out_prefill.detach().clone()
                        if _spark_active_page_dump_enabled()
                        else None
                    )

                    if spark_tensor_trace_should_emit(
                        "flashinfer_wrapper_prefill_pre", layer_name
                    ):
                        spark_tensor_trace(
                            "flashinfer_wrapper_prefill_pre",
                            {
                                "layer_name": layer_name,
                                "kv_cache_dtype": str(self.kv_cache_dtype),
                                "is_kvcache_nvfp4": bool(self.is_kvcache_nvfp4),
                                "use_fa2_nvfp4_kv": bool(self.use_fa2_nvfp4_kv),
                                "needs_fp8_out": bool(needs_fp8_out_prefill),
                                "uses_contig_out": bool(uses_contig_out_prefill),
                                "wrapper_type": type(prefill_wrapper).__name__,
                                "window_left": int(self.window_left),
                                "head_dim": int(self.head_size),
                                "num_q_heads": int(self.num_heads),
                                "num_kv_heads": int(self.num_kv_heads),
                                "num_prefill_tokens": int(num_prefill_tokens),
                                "num_decode_tokens": int(num_decode_tokens),
                                "k_scale": _spark_trace_scalar(layer._k_scale_float),
                                "v_scale": _spark_trace_scalar(layer._v_scale_float),
                                "query_last": spark_trace_last_token_summary(
                                    prefill_query
                                ),
                                "kv_cache_arg": _spark_tensor_trace_tuple_payload(
                                    kv_cache_permute
                                ),
                                "kv_cache_sf": _spark_tensor_trace_tuple_payload(
                                    kv_cache_sf
                                ),
                                "out_before": spark_trace_last_token_summary(
                                    out_prefill
                                ),
                                "output_view": _spark_tensor_trace_view_payload(output),
                                "out_target_view": _spark_tensor_trace_view_payload(
                                    out_prefill_target
                                ),
                                "out_arg_view": _spark_tensor_trace_view_payload(
                                    out_prefill
                                ),
                            },
                        )

                    if self.vo_split > 1:
                        if self.is_kvcache_nvfp4:
                            assert isinstance(kv_cache_permute, tuple)
                            assert isinstance(kv_cache_sf, tuple)
                        self._run_vo_split_prefill(
                            prefill_wrapper,
                            prefill_query,
                            kv_cache_permute,
                            kv_cache_sf,
                            out_prefill,
                            k_scale=layer._k_scale_float,
                            v_scale=layer._v_scale_float,
                        )
                    else:
                        prefill_wrapper.run(
                            prefill_query,
                            kv_cache_permute,
                            k_scale=layer._k_scale_float,
                            v_scale=layer._v_scale_float,
                            out=out_prefill,
                            kv_cache_sf=kv_cache_sf,
                        )

                    if (
                        self.is_kvcache_nvfp4
                        and self.use_fa2_nvfp4_kv
                        and isinstance(prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper)
                        and spark_tensor_trace_should_emit(
                            "flashinfer_wrapper_prefill_plan_run", layer_name
                        )
                    ):
                        spark_tensor_trace(
                            "flashinfer_wrapper_prefill_plan_run",
                            {
                                "layer_name": layer_name,
                                "kv_cache_dtype": str(self.kv_cache_dtype),
                                "window_left": int(self.window_left),
                                "head_dim": int(self.head_size),
                                "num_q_heads": int(self.num_heads),
                                "num_kv_heads": int(self.num_kv_heads),
                                "num_prefill_tokens": int(num_prefill_tokens),
                                "num_decode_tokens": int(num_decode_tokens),
                                "k_scale": _spark_trace_scalar(layer._k_scale_float),
                                "v_scale": _spark_trace_scalar(layer._v_scale_float),
                                "query": _spark_tensor_trace_view_payload(
                                    prefill_query
                                ),
                                "kv_cache_arg": _spark_tensor_trace_tuple_payload(
                                    kv_cache_permute
                                ),
                                "kv_cache_sf": _spark_tensor_trace_tuple_payload(
                                    kv_cache_sf
                                ),
                                "out_arg": _spark_tensor_trace_view_payload(
                                    out_prefill
                                ),
                                "wrapper_plan": _spark_trace_prefill_wrapper_payload(
                                    prefill_wrapper
                                ),
                                "flashinfer_extra_cudaflags": os.environ.get(
                                    "FLASHINFER_EXTRA_CUDAFLAGS", ""
                                ),
                            },
                        )

                    if (
                        self.is_kvcache_nvfp4
                        and self.use_fa2_nvfp4_kv
                        # The replay re-plans with symmetric head dims, which
                        # the VO-split wrapper deliberately does not use.
                        and self.vo_split == 1
                        and isinstance(prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper)
                        and isinstance(kv_cache_permute, tuple)
                    ):
                        self._spark_nvfp4_prefill_fresh_wrapper_replay(
                            layer=layer,
                            layer_name=layer_name,
                            live_wrapper=prefill_wrapper,
                            prefill_query=prefill_query,
                            kv_cache_permute=kv_cache_permute,
                            kv_cache_sf=kv_cache_sf
                            if isinstance(kv_cache_sf, tuple)
                            else None,
                            live_out=out_prefill,
                        )

                    if (
                        dump_out_before is not None
                        and isinstance(prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper)
                    ):
                        _spark_active_page_dump(
                            event="prefill",
                            layer_name=layer_name,
                            query=prefill_query,
                            out_before=dump_out_before,
                            out_after=out_prefill,
                            kv_data=kv_cache_permute
                            if isinstance(kv_cache_permute, tuple)
                            else None,
                            kv_scales=kv_cache_sf
                            if isinstance(kv_cache_sf, tuple)
                            else None,
                            wrapper=prefill_wrapper,
                            k_scale=layer._k_scale_float,
                            v_scale=layer._v_scale_float,
                            window_left=self.window_left,
                            num_prefill_tokens=num_prefill_tokens,
                            num_decode_tokens=num_decode_tokens,
                        )

                    if spark_tensor_trace_should_emit(
                        "flashinfer_wrapper_prefill_post", layer_name
                    ):
                        spark_tensor_trace(
                            "flashinfer_wrapper_prefill_post",
                            {
                                "layer_name": layer_name,
                                "kv_cache_dtype": str(self.kv_cache_dtype),
                                "is_kvcache_nvfp4": bool(self.is_kvcache_nvfp4),
                                "use_fa2_nvfp4_kv": bool(self.use_fa2_nvfp4_kv),
                                "needs_fp8_out": bool(needs_fp8_out_prefill),
                                "uses_contig_out": bool(uses_contig_out_prefill),
                                "wrapper_type": type(prefill_wrapper).__name__,
                                "window_left": int(self.window_left),
                                "head_dim": int(self.head_size),
                                "num_q_heads": int(self.num_heads),
                                "num_kv_heads": int(self.num_kv_heads),
                                "num_prefill_tokens": int(num_prefill_tokens),
                                "num_decode_tokens": int(num_decode_tokens),
                                "k_scale": _spark_trace_scalar(layer._k_scale_float),
                                "v_scale": _spark_trace_scalar(layer._v_scale_float),
                                "out_after": spark_trace_last_token_summary(
                                    out_prefill
                                ),
                                "out_target_view": _spark_tensor_trace_view_payload(
                                    out_prefill_target
                                ),
                                "out_arg_view": _spark_tensor_trace_view_payload(
                                    out_prefill
                                ),
                            },
                        )

                    if needs_fp8_out_prefill:
                        output[
                            num_decode_tokens : num_decode_tokens + num_prefill_tokens
                        ].copy_(out_prefill.to(output.dtype))
                    elif uses_contig_out_prefill:
                        out_prefill_target.copy_(out_prefill)
            else:
                assert isinstance(attn_metadata.prefill, TRTLLMPrefill)
                # prefill_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                prefill_query = prefill_query.contiguous()
                prefill_query = canonicalize_singleton_dim_strides(prefill_query)
                workspace_buffer = _get_trtllm_gen_workspace_buffer()
                block_tables_prefill = attn_metadata.prefill.block_tables
                seq_lens_prefill = attn_metadata.prefill.seq_lens

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_kv_cache_layout() == "HND"
                assert is_strictly_contiguous(prefill_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_prefill)
                assert is_strictly_contiguous(seq_lens_prefill)

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[num_decode_tokens:],
                        scale=output_block_scale,
                        scale_start_index=num_decode_tokens,
                        original_shape=prefill_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[num_decode_tokens:]

                # SM100 TRTLLM NVFP4 only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_prefill_tokens]

                prefill_kv_block_scales = None
                if self.is_kvcache_nvfp4:
                    # NVFP4 trtllm-gen kernel requires FP8 query.
                    assert attn_metadata.q_data_type == FP8_DTYPE, (
                        "NVFP4 KV cache requires FP8 quantized queries for "
                        "trtllm-gen prefill. Set "
                        "disable_flashinfer_q_quantization=False."
                    )
                    mock_kv_cache = nvfp4_kv_data
                    mock_block_table = block_tables_prefill
                    prefill_kv_block_scales = nvfp4_kv_block_scales
                elif (
                    attn_metadata.q_data_type != FP8_DTYPE
                    and self.kv_cache_dtype.startswith("fp8")
                ):
                    # TRTLLM prefill attention does not support BF16 Q
                    # and fp8 kv cache. So to enable prefill attention
                    # with fp8 kv cache, we can construct a mock block
                    # and mock kv cache with BF16 KV involved in the prefill
                    #
                    kv_cache_permute = canonicalize_singleton_dim_strides(
                        kv_cache_permute
                    )
                    kv_strides = kv_cache_permute.stride()
                    assert (
                        kv_strides[-1] == 1
                        and kv_strides[-2] == kv_cache_permute.shape[-1]
                    ), (
                        "KV cache inner dims (block_size, head_size) must be "
                        f"contiguous, got strides {kv_strides}"
                    )
                    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
                        kv_cache_permute,
                        block_tables_prefill,
                        layer._k_scale,
                        layer._v_scale,
                        attn_metadata.q_data_type,
                    )
                else:
                    mock_kv_cache = kv_cache_permute
                    mock_block_table = block_tables_prefill

                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=mock_kv_cache,
                    workspace_buffer=workspace_buffer,
                    block_tables=mock_block_table,
                    seq_lens=seq_lens_prefill,
                    max_q_len=attn_metadata.prefill.max_q_len,
                    max_kv_len=attn_metadata.prefill.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    batch_size=attn_metadata.num_prefills,
                    cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
                    cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    kv_cache_sf=prefill_kv_block_scales,
                )

                if needs_fp8_out:
                    output[
                        num_decode_tokens : num_decode_tokens + num_prefill_tokens
                    ].copy_(out[:num_prefill_tokens].to(output.dtype))

        if num_decode_tokens > 0:
            assert self.vo_split == 1, (
                "NVFP4 VO split routes decodes through the prefill wrapper; "
                "no tokens should reach the decode pathway."
            )
            decode_query = query[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens

            if not decode_use_trtllm:
                assert isinstance(attn_metadata.decode, FIDecode)
                decode_wrapper = attn_metadata.decode.wrapper
                assert decode_wrapper is not None
                assert decode_wrapper._window_left == self.window_left
                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)
                assert decode_wrapper._sm_scale == self.scale

                if self.is_kvcache_nvfp4:
                    kv_cache_permute = nvfp4_kv_data
                kv_cache_sf = nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None

                # SM100 TRTLLM NVFP4 only supports FP8 output. The SM12x FA2
                # path writes model dtype directly.
                needs_fp8_out = (
                    self.is_kvcache_nvfp4
                    and not self.use_fa2_nvfp4_kv
                    and output.dtype != FP8_DTYPE
                )
                if needs_fp8_out:
                    out_decode = self._nvfp4_fp8_out[:num_decode_tokens]
                else:
                    out_decode = output[:num_decode_tokens]

                if use_dcp:
                    decode_query = get_dcp_group().all_gather(
                        decode_query.contiguous(), dim=-2
                    )
                    output_tmp = torch.empty_like(decode_query)
                    lse = torch.empty(
                        (decode_query.size(0), decode_query.size(1)),
                        dtype=torch.float32,
                        device=decode_query.device,
                    )
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=output_tmp,
                        lse=lse,
                        return_lse=True,
                        kv_cache_sf=kv_cache_sf,
                    )
                    output[:num_decode_tokens] = self.dcp_combine(
                        output_tmp,
                        lse,
                        get_dcp_group(),
                    )
                else:
                    if spark_tensor_trace_should_emit(
                        "flashinfer_wrapper_decode_pre", layer_name
                    ):
                        spark_tensor_trace(
                            "flashinfer_wrapper_decode_pre",
                            {
                                "layer_name": layer_name,
                                "kv_cache_dtype": str(self.kv_cache_dtype),
                                "is_kvcache_nvfp4": bool(self.is_kvcache_nvfp4),
                                "use_fa2_nvfp4_kv": bool(self.use_fa2_nvfp4_kv),
                                "needs_fp8_out": bool(needs_fp8_out),
                                "wrapper_type": type(decode_wrapper).__name__,
                                "window_left": int(self.window_left),
                                "head_dim": int(self.head_size),
                                "num_q_heads": int(self.num_heads),
                                "num_kv_heads": int(self.num_kv_heads),
                                "num_decode_tokens": int(num_decode_tokens),
                                "k_scale": _spark_trace_scalar(layer._k_scale_float),
                                "v_scale": _spark_trace_scalar(layer._v_scale_float),
                                "query_last": spark_trace_last_token_summary(
                                    decode_query
                                ),
                                "kv_cache_arg": _spark_tensor_trace_tuple_payload(
                                    kv_cache_permute
                                ),
                                "kv_cache_sf": _spark_tensor_trace_tuple_payload(
                                    kv_cache_sf
                                ),
                                "out_before": spark_trace_last_token_summary(
                                    out_decode
                                ),
                            },
                        )

                    decode_wrapper.run(
                        decode_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=out_decode,
                        kv_cache_sf=kv_cache_sf,
                    )

                    if spark_tensor_trace_should_emit(
                        "flashinfer_wrapper_decode_post", layer_name
                    ):
                        spark_tensor_trace(
                            "flashinfer_wrapper_decode_post",
                            {
                                "layer_name": layer_name,
                                "kv_cache_dtype": str(self.kv_cache_dtype),
                                "is_kvcache_nvfp4": bool(self.is_kvcache_nvfp4),
                                "use_fa2_nvfp4_kv": bool(self.use_fa2_nvfp4_kv),
                                "needs_fp8_out": bool(needs_fp8_out),
                                "wrapper_type": type(decode_wrapper).__name__,
                                "window_left": int(self.window_left),
                                "head_dim": int(self.head_size),
                                "num_q_heads": int(self.num_heads),
                                "num_kv_heads": int(self.num_kv_heads),
                                "num_decode_tokens": int(num_decode_tokens),
                                "k_scale": _spark_trace_scalar(layer._k_scale_float),
                                "v_scale": _spark_trace_scalar(layer._v_scale_float),
                                "out_after": spark_trace_last_token_summary(
                                    out_decode
                                ),
                            },
                        )

                if needs_fp8_out:
                    output[:num_decode_tokens].copy_(out_decode.to(output.dtype))
            else:
                assert isinstance(attn_metadata.decode, TRTLLMDecode)
                # decode_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                decode_query = decode_query.contiguous()
                decode_query = canonicalize_singleton_dim_strides(decode_query)
                workspace_buffer = _get_trtllm_gen_workspace_buffer()
                block_tables_decode = attn_metadata.decode.block_tables
                seq_lens_decode = attn_metadata.decode.seq_lens

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_kv_cache_layout() == "HND"
                assert is_strictly_contiguous(decode_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_decode)
                assert is_strictly_contiguous(seq_lens_decode)
                kv_cache_permute = canonicalize_singleton_dim_strides(kv_cache_permute)
                kv_strides = kv_cache_permute.stride()
                assert (
                    kv_strides[-1] == 1 and kv_strides[-2] == kv_cache_permute.shape[-1]
                ), (
                    "KV cache inner dims (block_size, head_size) must be "
                    f"contiguous, got strides {kv_strides}"
                )

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[:num_decode_tokens],
                        scale=output_block_scale,
                        scale_start_index=0,
                        original_shape=decode_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[:num_decode_tokens]

                # SM100 TRTLLM NVFP4 only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = self.is_kvcache_nvfp4 and output.dtype != FP8_DTYPE
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_decode_tokens]

                if num_decode_tokens % attn_metadata.num_decodes != 0:
                    # This gets triggered when the dummy_run forces
                    # attention to be initialized with q_len = 0
                    q_len_per_req = 1
                else:
                    q_len_per_req = num_decode_tokens // attn_metadata.num_decodes

                trtllm_batch_decode_with_kv_cache(
                    query=decode_query,
                    kv_cache=(
                        nvfp4_kv_data if self.is_kvcache_nvfp4 else kv_cache_permute
                    ),
                    workspace_buffer=workspace_buffer,
                    block_tables=block_tables_decode,
                    seq_lens=seq_lens_decode,
                    max_seq_len=attn_metadata.decode.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    q_len_per_req=q_len_per_req,
                    kv_cache_sf=(
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    ),
                )

                if needs_fp8_out:
                    output[:num_decode_tokens].copy_(out.to(output.dtype))
        if spark_tensor_trace_should_emit("flashinfer_attn_output", layer_name):
            spark_tensor_trace(
                "flashinfer_attn_output",
                {
                    "layer_name": layer_name,
                    "kv_cache_dtype": str(self.kv_cache_dtype),
                    "is_kvcache_nvfp4": bool(self.is_kvcache_nvfp4),
                    "use_fa2_nvfp4_kv": bool(self.use_fa2_nvfp4_kv),
                    "window_left": int(self.window_left),
                    "head_dim": int(self.head_size),
                    "num_q_heads": int(self.num_heads),
                    "num_kv_heads": int(self.num_kv_heads),
                    "num_actual_tokens": int(num_actual_tokens),
                    "num_decodes": int(attn_metadata.num_decodes),
                    "num_decode_tokens": int(num_decode_tokens),
                    "num_prefills": int(attn_metadata.num_prefills),
                    "num_prefill_tokens": int(num_prefill_tokens),
                    "output_last": spark_trace_last_token_summary(output),
                },
            )
        return output_padded

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            k_cache = kv_cache[:, 0]
            v_cache = kv_cache[:, 1]
            layer_name = _spark_kv_trace_layer_name(layer)
            if _spark_kv_trace_should_emit("kv_write_pre", layer_name=layer_name):
                _spark_kv_trace(
                    "kv_write_pre",
                    {
                        "layer_name": layer_name,
                        "kv_cache_dtype": str(self.kv_cache_dtype),
                        "key_shape": list(key.shape),
                        "key_stride": list(key.stride()),
                        "key_dtype": str(key.dtype),
                        "value_shape": list(value.shape),
                        "value_stride": list(value.stride()),
                        "value_dtype": str(value.dtype),
                        "kv_cache_shape": list(kv_cache.shape),
                        "kv_cache_stride": list(kv_cache.stride()),
                        "kv_cache_dtype_actual": str(kv_cache.dtype),
                        "k_cache_shape": list(k_cache.shape),
                        "k_cache_stride": list(k_cache.stride()),
                        "v_cache_shape": list(v_cache.shape),
                        "v_cache_stride": list(v_cache.stride()),
                        "slot_mapping_shape": list(slot_mapping.shape),
                        "slot_mapping_head": _spark_kv_trace_tensor_head(
                            slot_mapping
                        ),
                        "head_dim": int(self.head_size),
                        "num_kv_heads": int(self.num_kv_heads),
                    },
                )
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
            if self.is_kvcache_nvfp4 and _spark_kv_trace_should_emit(
                "kv_write_post_nvfp4", layer_name=layer_name
            ):
                stride_order = FlashInferBackend.get_kv_cache_stride_order()
                kv_cache_permute = kv_cache.permute(*stride_order)
                data_views, scale_views = nvfp4_kv_cache_split_views(kv_cache_permute)
                page_size = int(kv_cache_permute.shape[2])
                _spark_kv_trace(
                    "kv_write_post_nvfp4",
                    {
                        "layer_name": layer_name,
                        "kv_cache_dtype": str(self.kv_cache_dtype),
                        "kv_cache_shape": list(kv_cache.shape),
                        "kv_cache_stride": list(kv_cache.stride()),
                        "kv_cache_permute_shape": list(kv_cache_permute.shape),
                        "kv_cache_permute_stride": list(kv_cache_permute.stride()),
                        "data_views": _spark_kv_trace_views_info(data_views),
                        "scale_views": _spark_kv_trace_views_info(scale_views),
                        "slot_mapping_head": _spark_kv_trace_tensor_head(
                            slot_mapping
                        ),
                        "slot_samples": _spark_kv_trace_slot_samples(
                            data_views,
                            scale_views,
                            slot_mapping,
                            page_size,
                        ),
                        "head_dim": int(self.head_size),
                        "num_kv_heads": int(self.num_kv_heads),
                    },
                )


def fast_plan_decode(
    self,  # decode wrapper
    indptr_cpu: torch.Tensor,
    indices: torch.Tensor,
    last_page_len_cpu: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: float | None = None,
    q_data_type: str | torch.dtype | None = "float16",
    kv_data_type: str | torch.dtype | None = None,
    o_data_type: str | torch.dtype | None = None,
    data_type: str | torch.dtype | None = None,
    sm_scale: float | None = None,
    rope_scale: float | None = None,
    rope_theta: float | None = None,
    non_blocking: bool = True,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> None:
    """
    A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for
    cudagraph capture/replay, while the no cudagraph version turns back
    to the original plan.
    using original plan after passing host-side buffers:
    - only host-to-device copy of indptr and last_page_len buffers
    Modifications for cudagraph:
    - only host-to-device copy of indptr and last_page_len buffers.
    - avoid device-to-device copy of indices buffer.

    Part of the code get inspiration from the original plan from FlashInfer repo
    and the implementation of fast_decode_plan for FlashInfer in SGlang repo.
    """
    # Warm up with the original plan if it is first call, and always run the
    # original plan if we run for dynamic shape. For fixed shape (cudagraph),
    # this warm up is to generate the _cached_module for the decode wrapper.
    if not self.is_cuda_graph_enabled or getattr(self, "vllm_first_call", True):
        self.plan(
            indptr=indptr_cpu,
            indices=indices,
            last_page_len=last_page_len_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            block_tables=None,
            seq_lens=None,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self.vllm_first_call = False
        return

    assert self.is_cuda_graph_enabled, "Should be cudagraph only here"

    fast_decode_plan(
        self,
        indptr=indptr_cpu,
        indices=indices,
        last_page_len=last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        data_type=data_type,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        non_blocking=non_blocking,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )
