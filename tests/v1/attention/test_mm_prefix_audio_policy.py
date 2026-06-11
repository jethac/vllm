# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static (no-GPU) AUDIO-policy tests for mm-prefix bidirectional masking.

spark-hijinks campaign, Amendment 5 (OVERNIGHT_LADDER_PLAN_20260612):
Gemma 4 E2B/E4B/12B carry audio encoders; the Triton-retirement mm flip
must not change AUDIO masking semantics.

Authoritative policy (HF transformers Gemma4 reference implementation):

- ``processing_utils.create_mm_token_type_ids``: text=0, image=1,
  video=2, audio=3.
- ``modeling_gemma4.get_block_sequence_ids_for_mask``:
  ``is_vision = (type == 1) | (type == 2)``; everything else maps to
  block ``-1`` = strictly causal. AUDIO soft tokens are therefore
  NEVER bidirectional in the language model (the audio tower is
  bidirectional internally, before the LM).

vLLM upstream implements the same policy by excluding audio
mm_features from ``mm_req_doc_ranges`` at the source (gpu_model_runner;
introduced with Gemma4 Unified support, PR #44429, reaffirmed in
PR #42175). The FlashInfer mm-prefix custom-mask path consumes ONLY
``mm_req_doc_ranges``, so audio correctness is inherited from that
exclusion — these tests pin it explicitly per modality so a regression
at either level (range source or builder span filter) is caught
statically.

Everything runs on CPU; no CUDA required.
"""

import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
from vllm.v1.worker.gpu_model_runner import mm_prefix_doc_ranges_for_request

GEMMA4_SLIDING_WINDOW = 512


def _feat(
    modality: str,
    offset: int,
    length: int,
    is_embed: torch.Tensor | None = None,
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality=modality,
        identifier=f"{modality}-{offset}-{length}",
        mm_position=PlaceholderRange(
            offset=offset, length=length, is_embed=is_embed
        ),
    )


class TestMMPrefixRangeSourcePolicy:
    """Range source: which modalities produce bidirectional doc ranges."""

    def test_image_produces_range(self):
        ranges = mm_prefix_doc_ranges_for_request(
            [_feat("image", 4, 64)], GEMMA4_SLIDING_WINDOW
        )
        assert ranges == [(4, 67)]

    def test_video_produces_range(self):
        # HF policy: video (mm_token_type_id 2) is "vision" too.
        ranges = mm_prefix_doc_ranges_for_request(
            [_feat("video", 10, 128)], GEMMA4_SLIDING_WINDOW
        )
        assert ranges == [(10, 137)]

    def test_audio_produces_no_range(self):
        # THE Amendment-5 policy cell: audio soft tokens stay strictly
        # causal in the LM; an audio feature must NEVER contribute a
        # bidirectional range, on any layer group.
        ranges = mm_prefix_doc_ranges_for_request(
            [_feat("audio", 4, 64)], GEMMA4_SLIDING_WINDOW
        )
        assert ranges == []

    def test_audio_skipped_even_when_span_fits_window(self):
        # The skip is a modality policy, not a window-guard side effect.
        ranges = mm_prefix_doc_ranges_for_request(
            [_feat("audio", 0, 8)], GEMMA4_SLIDING_WINDOW
        )
        assert ranges == []

    def test_audio_skipped_without_sliding_window(self):
        # Window guard disabled (None) must not re-admit audio.
        ranges = mm_prefix_doc_ranges_for_request([_feat("audio", 4, 64)], None)
        assert ranges == []

    def test_mixed_audio_image_keeps_only_image_ranges(self):
        feats = [
            _feat("audio", 2, 30),
            _feat("image", 40, 64),
            _feat("audio", 110, 16),
        ]
        ranges = mm_prefix_doc_ranges_for_request(feats, GEMMA4_SLIDING_WINDOW)
        assert ranges == [(40, 103)]

    def test_image_span_longer_than_sliding_window_dropped(self):
        # Window guard (upstream PR #40534): spans longer than the text
        # sliding window are dropped at the source.
        ranges = mm_prefix_doc_ranges_for_request(
            [_feat("image", 0, GEMMA4_SLIDING_WINDOW + 1)],
            GEMMA4_SLIDING_WINDOW,
        )
        assert ranges == []

    def test_image_span_equal_to_sliding_window_kept(self):
        ranges = mm_prefix_doc_ranges_for_request(
            [_feat("image", 0, GEMMA4_SLIDING_WINDOW)], GEMMA4_SLIDING_WINDOW
        )
        assert ranges == [(0, GEMMA4_SLIDING_WINDOW - 1)]

    def test_is_embed_mask_split_ranges_image_only(self):
        # extract_embeds_range() fidelity: a masked image placeholder
        # yields split ranges; the same mask on audio yields nothing.
        is_embed = torch.tensor(
            [False, True, False, True, True], dtype=torch.bool
        )
        image_ranges = mm_prefix_doc_ranges_for_request(
            [_feat("image", 2, 5, is_embed=is_embed)], GEMMA4_SLIDING_WINDOW
        )
        assert image_ranges == [(3, 3), (5, 6)]
        audio_ranges = mm_prefix_doc_ranges_for_request(
            [_feat("audio", 2, 5, is_embed=is_embed)], GEMMA4_SLIDING_WINDOW
        )
        assert audio_ranges == []

    def test_no_features_empty(self):
        assert mm_prefix_doc_ranges_for_request([], GEMMA4_SLIDING_WINDOW) == []


def _spans(
    seq_lens: list[int],
    query_lens: list[int],
    mm_req_doc_ranges: dict[int, list[tuple[int, int]]] | None,
    num_decodes: int = 0,
):
    """Run FlashInferMetadataBuilder._mm_prefix_prefill_spans statically.

    The method reads only CommonAttentionMetadata (no builder state), so
    it can be exercised unbound on CPU without a FlashInfer install
    being exercised.
    """
    meta = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=torch.device("cpu"),
    )
    meta.mm_req_doc_ranges = mm_req_doc_ranges
    num_prefills = len(seq_lens) - num_decodes
    return FlashInferMetadataBuilder._mm_prefix_prefill_spans(
        None, meta, num_decodes, num_prefills
    )


class TestFlashInferAudioSpanFilter:
    """Builder span filter: audio-only requests classify all-plain."""

    def test_no_ranges_returns_none(self):
        assert _spans([64], [64], None) is None

    def test_audio_only_request_returns_none(self):
        # What the model runner produces for an audio-only mm request:
        # an entry with NO ranges. The builder must take the legacy
        # scalar-causal path (byte-identical regression guarantee).
        assert _spans([64], [64], {0: []}) is None

    def test_image_request_returns_spans(self):
        spans = _spans([64], [64], {0: [(4, 20)]})
        assert spans == [[(4, 20)]]

    def test_mixed_audio_image_batch(self):
        # req0 audio-only (empty ranges), req1 image: only req1 carries
        # a span; req0 lands in the plain causal group.
        spans = _spans([64, 64], [64, 64], {0: [], 1: [(8, 30)]})
        assert spans == [[], [(8, 30)]]

    def test_all_audio_batch_returns_none(self):
        assert _spans([64, 64], [64, 64], {0: [], 1: []}) is None

    def test_image_span_fully_in_context_filtered(self):
        # Span entirely inside the computed context (e.g. decode-shaped
        # qo_len==1 rows under the VO split): no custom mask needed.
        spans = _spans([65], [1], {0: [(4, 20)]})
        assert spans is None

    def test_degenerate_start_eq_end_filtered(self):
        # mm_req_doc_ranges convention: valid iff start < end.
        spans = _spans([64], [64], {0: [(5, 5)]})
        assert spans is None


class TestGemma4AudioRoutingUnchanged:
    """Audio presence does not change mm-prefix classification."""

    def test_vision_policy_checkpoint_is_mm_prefix_lm(self):
        # Gemma4 audio-carrying checkpoints (E2B/E4B/12B) still declare
        # use_bidirectional_attention='vision'; audio_config presence is
        # irrelevant to is_mm_prefix_lm (routing identical with/without
        # audio inputs; the audio policy is enforced at the range
        # source, not the selector).
        from types import SimpleNamespace

        from vllm.transformers_utils.model_arch_config_convertor import (
            Gemma4ModelArchConfigConvertor,
        )

        text_cfg = SimpleNamespace(use_bidirectional_attention="vision")
        hf_cfg = SimpleNamespace(audio_config=SimpleNamespace())
        conv = Gemma4ModelArchConfigConvertor(hf_cfg, text_cfg)
        assert conv.is_mm_prefix_lm() is True

        text_cfg_none = SimpleNamespace(use_bidirectional_attention=None)
        conv_none = Gemma4ModelArchConfigConvertor(hf_cfg, text_cfg_none)
        assert conv_none.is_mm_prefix_lm() is False
