"""NVFP4 KV calibrated global-scale loader (productionized calibration policy).

Drop-in module for vLLM (place at `vllm/nvfp4_kv_calib.py`). The attention layer init calls
`calibrated_kv_scales(hf_config)` and, when nvfp4 KV is active on the FA2 path and a calibration
matches, sets `layer._k_scale`/`_v_scale` from it instead of the uncalibrated `1.0` placeholder.

KEYING: by **architecture signature**, NOT HF model name. The optimal nvfp4-KV global scale is a
property of the architecture + shape (head_dim, hidden_size, layer count, kv heads) — it is invariant
across fine-tunes / abliterations / merges / re-uploads of the same base, which all share the config.
So `google/gemma-4-12b-it` and any derivative of it resolve to the SAME calibration. (Name-matching
would silently miss every variant — a bad design.)

Calibration source (env `VLLM_NVFP4_KV_CALIB`):
  - a directory -> looks up "<arch_signature>.json" inside it, or
  - a single JSON file -> applied iff its "arch_signature" matches (or it omits one, = wildcard).
JSON shape: {"arch_signature": "...", "k_scale": 0.1, "v_scale": 0.1, ...}.
Produced offline by docs/vast_anchor/nvfp4_kv_calibrate_nll.py.

No-op (returns None) when the env is unset or nothing matches — additive, opt-in.
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any, Optional, Tuple


@functools.lru_cache(maxsize=16)
def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _cfg_get(hf_config: Any, key: str) -> Any:
    """Read key from hf_config, falling back to a nested text_config (multimodal Gemma)."""
    v = getattr(hf_config, key, None)
    if v is None:
        tc = getattr(hf_config, "text_config", None)
        if tc is not None:
            v = getattr(tc, key, None)
    return v


def arch_signature(hf_config: Any) -> Optional[str]:
    """Stable, variant-invariant key from the architecture + shape that determine KV statistics."""
    if hf_config is None:
        return None
    archs = getattr(hf_config, "architectures", None) or [None]
    arch = archs[0] or _cfg_get(hf_config, "model_type") or "unknown"
    parts = [
        str(arch),
        f"L{_cfg_get(hf_config, 'num_hidden_layers')}",
        f"H{_cfg_get(hf_config, 'hidden_size')}",
        f"D{_cfg_get(hf_config, 'head_dim')}",
        f"KV{_cfg_get(hf_config, 'num_key_value_heads')}",
    ]
    return "-".join(parts)


def calibrated_kv_scales(hf_config: Any) -> Optional[Tuple[float, float]]:
    """Return (k_scale, v_scale) for this architecture, or None if nothing is configured/matches."""
    src = os.environ.get("VLLM_NVFP4_KV_CALIB")
    if not src:
        return None
    sig = arch_signature(hf_config)
    if sig is None:
        return None
    if os.path.isdir(src):
        d = _load_json(os.path.join(src, sig + ".json"))
    else:
        d = _load_json(src)
        # single file: apply iff its signature matches (missing signature = wildcard).
        if d and d.get("arch_signature") and d["arch_signature"] != sig:
            return None
    if not d:
        return None
    k, v = d.get("k_scale"), d.get("v_scale")
    if k is None or v is None:
        return None
    try:
        return float(k), float(v)
    except (TypeError, ValueError):
        return None
