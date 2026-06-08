# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inactive-by-default tensor summaries for Spark/GB10 debugging."""

import json
import os
from typing import Any

import torch

_SPARK_TENSOR_TRACE_COUNTS: dict[tuple[str, str], int] = {}


def spark_tensor_trace_enabled() -> bool:
    return os.environ.get("VLLM_SPARK_GEMMA_TENSOR_TRACE") == "1"


def _spark_tensor_trace_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def spark_tensor_trace_wants_layer(layer_name: str | None) -> bool:
    raw_filter = os.environ.get("VLLM_SPARK_GEMMA_TENSOR_TRACE_LAYERS", "")
    filters = [item.strip() for item in raw_filter.split(",") if item.strip()]
    if not filters:
        return True
    candidate = layer_name or ""
    return any(item in candidate for item in filters)


def spark_tensor_trace_should_emit(event: str, layer_name: str | None) -> bool:
    if not spark_tensor_trace_enabled():
        return False
    if not spark_tensor_trace_wants_layer(layer_name):
        return False
    limit = _spark_tensor_trace_int("VLLM_SPARK_GEMMA_TENSOR_TRACE_LIMIT", 4)
    if limit == 0:
        return False
    key = (event, layer_name or "<none>")
    count = _SPARK_TENSOR_TRACE_COUNTS.get(key, 0)
    if count >= limit:
        return False
    _SPARK_TENSOR_TRACE_COUNTS[key] = count + 1
    return True


def _tensor_head(
    tensor: torch.Tensor,
    limit: int,
) -> list[int | float | bool | str]:
    if limit <= 0 or tensor.numel() == 0:
        return []
    flat = tensor.detach().reshape(-1)
    try:
        return [flat[i].cpu().item() for i in range(min(limit, flat.numel()))]
    except Exception as exc:
        return [f"<read_error:{type(exc).__name__}>"]


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "stride": list(detached.stride()),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        return summary
    try:
        values = detached.float()
        finite = torch.isfinite(values)
        summary["finite"] = int(finite.sum().cpu().item())
        if bool(finite.any().cpu().item()):
            finite_values = values[finite]
            summary.update(
                {
                    "min": float(finite_values.min().cpu().item()),
                    "max": float(finite_values.max().cpu().item()),
                    "mean": float(finite_values.mean().cpu().item()),
                    "rms": float(
                        torch.sqrt(torch.mean(finite_values * finite_values))
                        .cpu()
                        .item()
                    ),
                    "max_abs": float(finite_values.abs().max().cpu().item()),
                }
            )
    except Exception as exc:
        summary["stats_error"] = type(exc).__name__
    return summary


def spark_tensor_summary(
    tensor: torch.Tensor | None,
    *,
    include_values: bool = True,
    topk: int = 0,
) -> dict[str, Any] | None:
    if tensor is None:
        return None
    values_limit = _spark_tensor_trace_int(
        "VLLM_SPARK_GEMMA_TENSOR_TRACE_VALUES", 8
    )
    summary = _tensor_stats(tensor)
    if include_values:
        summary["head"] = _tensor_head(tensor, values_limit)
    if topk > 0 and tensor.numel() > 0:
        try:
            row = tensor.detach()
            if row.ndim > 1:
                row = row.reshape(-1, row.shape[-1])[-1]
            else:
                row = row.reshape(-1)
            k = min(topk, row.numel())
            top_values, top_indices = torch.topk(row.float(), k=k)
            summary["topk"] = [
                {
                    "token_id": int(index.cpu().item()),
                    "value": float(value.cpu().item()),
                }
                for value, index in zip(top_values, top_indices)
            ]
        except Exception as exc:
            summary["topk_error"] = type(exc).__name__
    return summary


def spark_trace_last_token_summary(
    tensor: torch.Tensor | None,
    *,
    include_values: bool = True,
    topk: int = 0,
) -> dict[str, Any] | None:
    if tensor is None:
        return None
    if tensor.ndim == 0 or tensor.shape[0] == 0:
        return spark_tensor_summary(tensor, include_values=include_values, topk=topk)
    return spark_tensor_summary(tensor[-1], include_values=include_values, topk=topk)


def spark_tensor_trace(event: str, payload: dict[str, Any]) -> None:
    if not spark_tensor_trace_enabled():
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
    path = os.environ.get("VLLM_SPARK_GEMMA_TENSOR_TRACE_FILE")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as trace_file:
                trace_file.write(line + "\n")
            return
        except OSError:
            pass
    print(f"Spark Gemma tensor trace: {line}", flush=True)
