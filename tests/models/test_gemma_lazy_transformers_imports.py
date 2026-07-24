# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import-robustness tests for the Gemma multimodal model modules.

The Gemma 3 / 3n / 4 multimodal modules reference symbols from version-gated
``transformers.models.<gemma*>`` submodules (and, for Gemma 3, a
torchvision-backed image processor). Those imports are deferred -- kept under
``TYPE_CHECKING`` for annotations and imported inside methods for runtime use --
so that importing the model module does not crash when the installed
transformers predates the model or torchvision is unavailable.

This matters because vLLM's architecture inspection imports a model module for
any registered architecture. An eager top-level import of a missing symbol
would raise ``ModuleNotFoundError`` at import time, breaking inspection of that
architecture (and any module that imports it) rather than failing only when the
model is actually used.

These tests do not require a GPU: they simulate the missing dependency with
``monkeypatch`` and assert the module still imports and registers, while the
processor helpers raise a clear error when actually invoked.
"""

import importlib
import sys

import pytest

from vllm.model_executor.models.registry import ModelRegistry

# (module import path, architecture registered in ``ModelRegistry``,
# transformers submodule that is absent on an older transformers install or
# without torchvision).
_IMPORT_CASES = [
    pytest.param(
        "vllm.model_executor.models.gemma3_mm",
        "Gemma3ForConditionalGeneration",
        "transformers.models.gemma3.image_processing_gemma3",
        id="gemma3_mm",
    ),
    pytest.param(
        "vllm.model_executor.models.gemma3n_mm",
        "Gemma3nForConditionalGeneration",
        "transformers.models.gemma3n",
        id="gemma3n_mm",
    ),
    pytest.param(
        "vllm.model_executor.models.gemma4_mm",
        "Gemma4ForConditionalGeneration",
        "transformers.models.gemma4",
        id="gemma4_mm",
    ),
]

# (module import path, transformers submodule made absent, callable that
# invokes the processor helper which imports that submodule lazily). Each
# callable reaches the deferred import before touching ``self.ctx``, so a
# ``None`` context is sufficient to exercise it.
_DEFERRED_CASES = [
    pytest.param(
        "vllm.model_executor.models.gemma4_mm",
        "transformers.models.gemma4",
        lambda module: module.Gemma4ProcessingInfo(ctx=None).get_hf_config(),
        id="gemma4_mm",
    ),
    pytest.param(
        "vllm.model_executor.models.gemma3n_mm",
        "transformers.models.gemma3n",
        lambda module: module.Gemma3nProcessingInfo(ctx=None).get_hf_config(),
        id="gemma3n_mm",
    ),
    pytest.param(
        "vllm.model_executor.models.gemma3_mm",
        "transformers.models.gemma3.processing_gemma3",
        lambda module: module.Gemma3ProcessingInfo(ctx=None).get_num_crops(
            image_width=1,
            image_height=1,
            processor=None,
            mm_kwargs={},
        ),
        id="gemma3_mm",
    ),
]


def _simulate_missing(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make ``import name`` fail, as on an older transformers or without
    torchvision.

    Args:
        monkeypatch: The pytest monkeypatch fixture (auto-reverts on teardown).
        name: Fully-qualified module that should appear absent.
    """
    for cached in list(sys.modules):
        if cached == name or cached.startswith(name + "."):
            monkeypatch.delitem(sys.modules, cached, raising=False)
    # A ``None`` entry in ``sys.modules`` makes ``import name`` raise
    # ``ImportError``, mirroring a genuinely missing module.
    monkeypatch.setitem(sys.modules, name, None)


@pytest.mark.parametrize("module_name, arch, missing", _IMPORT_CASES)
def test_module_imports_without_version_gated_transformers(
    module_name: str,
    arch: str,
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module imports and registers even when its version-gated
    transformers submodule is unavailable."""
    _simulate_missing(monkeypatch, missing)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    module = importlib.import_module(module_name)

    assert hasattr(module, arch)
    assert arch in ModelRegistry.get_supported_archs()


@pytest.mark.parametrize("module_name, missing, invoke", _DEFERRED_CASES)
def test_processor_use_raises_when_transformers_missing(
    module_name: str,
    missing: str,
    invoke,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using a processor helper surfaces the missing dependency as an
    ``ImportError`` at call time, not at module import time."""
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)

    _simulate_missing(monkeypatch, missing)

    with pytest.raises(ImportError):
        invoke(module)
