"""High-level loader for OmniRetriever-7B.

This module wraps the WAVE-7B backbone (a Qwen2.5-Omni extension with a BEATs
audio encoder) and applies the released LoRA adapter to produce text, video,
audio, and joint AVT embeddings in a single shared space.

Typical use::

    from omniretriever import OmniRetriever

    model = OmniRetriever.from_pretrained(
        base_model="<wave-7b hub id or local path>",
        adapter="YunzeLiu/OmniRetriever-7B",
        device="cuda",
        dtype="bfloat16",
    )
    z_text  = model.encode_text("a dog barking in the rain")
    z_video = model.encode_video("clip.mp4")
    z_audio = model.encode_audio("clip.wav")
    z_av    = model.encode_av("clip.mp4")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

logger = logging.getLogger(__name__)

# Default inference settings (match paper §3.5 and Appendix A).
DEFAULT_VIDEO_FRAMES = 8
DEFAULT_VIDEO_RESOLUTION = 224
DEFAULT_AUDIO_DURATION_SEC = 8
DEFAULT_AUDIO_SAMPLE_RATE = 16_000
DEFAULT_EMBED_DIM = 3584


@dataclass
class InferenceConfig:
    """Inference-time configuration.

    Attributes:
        video_max_frames: number of frames sampled per video (default 8).
        video_resolution: square pixel size each frame is resized to.
        audio_duration_sec: fixed-duration audio crop (centred).
        audio_sample_rate: target audio sample rate (Hz).
        embed_dim: dimensionality of the output embedding.
        normalize: whether to L2-normalize the output embedding.
        precision: numerical precision (one of ``"float32"``, ``"bfloat16"``,
            ``"float16"``).
        pin_video_resolution: whether to hold frames at ``video_resolution``
            instead of the pixel floor WAVE-7B ships, which rescales 224 px up
            to 336 px (576 visual tokens rather than 256). Off by default: the
            floor is the released configuration, and pinning cost ``v2t`` 0.087
            R@1 on a 300-record run. See
            ``omniretriever.inference.encode._video_kwargs``.
        append_turn_end: whether to close each prompt with the tokeniser's
            ``<|im_end|>``, as the training pipeline does. Added downstream; the
            fusion head pools a fixed final position, so without it the
            embedding is read out of a different token for every modality. Not
            applied on the ``av`` path, which cannot take a trailing token. See
            ``omniretriever.inference.encode._turn_end``.
        media_instruction: filler placed next to a media placeholder when the
            combination carries no caption. Added downstream to match the
            training recipe; set it to ``""`` to reproduce the released
            behaviour of sending a bare placeholder.
        instruction_paths: which encoders actually use that filler. It does not
            help everywhere. Measured on 300 records, adding it to the audio
            path lifted ``t2a`` from 0.110 to 0.197 and ``a2t`` from 0.103 to
            0.187, while adding it to the video path cost ``v2t`` 0.087 and
            ``v2at`` 0.077 for a 0.013 gain on ``t2v``. So ``video`` is out by
            default and ``audio`` is in. ``av`` carries both streams, so neither
            contrast covers it; it keeps the filler until someone measures it.
        duplicate_audio_tokens: whether to give each audio frame the second
            token slot the interleaved BEATs branch fills. Added downstream and
            defaulted **off**: Table S2 of the paper puts the joint AV forward
            at about 470 tokens, which the undoubled layout matches (465) and
            the doubled one does not (665). A 300-record A/B moved AVG-all by
            +0.004 R@1, i.e. nothing. See
            ``omniretriever.inference.encode._apply_beats_audio_slots``.
    """

    video_max_frames: int = DEFAULT_VIDEO_FRAMES
    video_resolution: int = DEFAULT_VIDEO_RESOLUTION
    audio_duration_sec: int = DEFAULT_AUDIO_DURATION_SEC
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    embed_dim: int = DEFAULT_EMBED_DIM
    normalize: bool = True
    precision: str = "bfloat16"
    append_turn_end: bool = True
    media_instruction: str = "Please describe the video."
    instruction_paths: tuple[str, ...] = ("audio", "av")
    duplicate_audio_tokens: bool = False
    pin_video_resolution: bool = False

    extra: dict = field(default_factory=dict)


class OmniRetriever:
    """Inference wrapper for OmniRetriever-7B (WAVE-7B + LoRA adapter).

    The class is intentionally thin: it holds the backbone, the LoRA adapter,
    the multimodal processor, and a unit-norm projection head, and exposes one
    ``encode_*`` method per modality combination. Heavy lifting (frame
    extraction, audio loading, tokenisation, batched forward) lives in the
    submodules under ``omniretriever.data`` and ``omniretriever.inference``.
    """

    def __init__(
        self,
        backbone,
        processor,
        config: InferenceConfig,
    ) -> None:
        self._backbone = backbone
        self._processor = processor
        self._config = config

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        base_model: str | os.PathLike,
        adapter: str | os.PathLike,
        device: str = "cuda",
        dtype: str = "bfloat16",
        config: InferenceConfig | None = None,
        duplicate_audio_tokens: bool = False,
        pin_video_resolution: bool = False,
        instruction_paths: Sequence[str] = ("audio", "av"),
        append_turn_end: bool = True,
    ) -> "OmniRetriever":
        """Load WAVE-7B and apply the OmniRetriever LoRA adapter.

        Args:
            base_model: path (or HF hub id) of the WAVE-7B backbone.
            adapter: path of the released LoRA adapter directory.
            device: PyTorch device string.
            dtype: precision (one of ``float32`` / ``bfloat16`` / ``float16``).
            config: optional :class:`InferenceConfig` override. When given, it
                is used as-is and the two flags below are ignored.
            duplicate_audio_tokens: see the field of the same name on
                :class:`InferenceConfig`.
            pin_video_resolution: see the field of the same name on
                :class:`InferenceConfig`.
            instruction_paths: see the field of the same name on
                :class:`InferenceConfig`.

        Returns:
            An :class:`OmniRetriever` instance ready for ``encode_*`` calls.
        """
        # Local imports avoid a heavy import-time dependency on transformers / peft
        # for users who only need the metrics module.
        from peft import PeftModel
        from transformers import AutoProcessor

        from omniretriever.models.wave import load_wave_backbone

        torch_dtype = _resolve_dtype(dtype)
        config = config or InferenceConfig(
            precision=dtype,
            duplicate_audio_tokens=duplicate_audio_tokens,
            pin_video_resolution=pin_video_resolution,
            instruction_paths=tuple(instruction_paths),
            append_turn_end=append_turn_end,
        )

        logger.info("Loading WAVE-7B backbone from %s", base_model)
        backbone = load_wave_backbone(base_model, torch_dtype=torch_dtype)

        logger.info("Applying OmniRetriever LoRA adapter from %s", adapter)
        backbone = PeftModel.from_pretrained(backbone, str(adapter))
        backbone = backbone.to(device).eval()

        processor = AutoProcessor.from_pretrained(str(base_model), trust_remote_code=True)

        return cls(backbone=backbone, processor=processor, config=config)

    # ------------------------------------------------------------------ #
    # Encoders                                                           #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def encode_text(self, text: str | Sequence[str]) -> torch.Tensor:
        """Encode one or more text strings into the shared embedding space.

        Args:
            text: a single string or a sequence of strings.

        Returns:
            Tensor of shape ``(N, D)`` with ``N == len(text)``.
        """
        from omniretriever.inference.encode import encode_text

        return encode_text(self._backbone, self._processor, text, self._config)

    @torch.inference_mode()
    def encode_video(self, video_path: str | Sequence[str]) -> torch.Tensor:
        """Encode one or more video files (video-only path, audio ignored)."""
        from omniretriever.inference.encode import encode_video

        return encode_video(self._backbone, self._processor, video_path, self._config)

    @torch.inference_mode()
    def encode_audio(self, audio_path: str | Sequence[str]) -> torch.Tensor:
        """Encode one or more audio files (audio-only path)."""
        from omniretriever.inference.encode import encode_audio

        return encode_audio(self._backbone, self._processor, audio_path, self._config)

    @torch.inference_mode()
    def encode_av(self, clip_path: str | Sequence[str]) -> torch.Tensor:
        """Encode one or more video files using both audio and video streams.

        ``clip_path`` should point to a container (MP4/MOV/WEBM) whose audio
        track is decoded jointly with the visual frames.
        """
        from omniretriever.inference.encode import encode_av

        return encode_av(self._backbone, self._processor, clip_path, self._config)

    # ------------------------------------------------------------------ #
    # Dual-modal encoders (added downstream, not in the original release) #
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def encode_tv(self, video_path, text) -> torch.Tensor:
        """Encode video and text jointly (the ``tv`` side of the benchmark).

        Added for this checkout; see the note above ``encode_tv`` in
        ``omniretriever.inference.encode`` for how the prompt is built and how
        it relates to the training-time recipe.
        """
        from omniretriever.inference.encode import encode_tv

        return encode_tv(self._backbone, self._processor, video_path, text, self._config)

    @torch.inference_mode()
    def encode_at(self, audio_path, text) -> torch.Tensor:
        """Encode audio and text jointly (the ``at`` side of the benchmark).

        Added for this checkout, same caveats as :meth:`encode_tv`.
        """
        from omniretriever.inference.encode import encode_at

        return encode_at(self._backbone, self._processor, audio_path, text, self._config)

    # ------------------------------------------------------------------ #
    # Accessors                                                          #
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> InferenceConfig:
        """Return the inference configuration."""
        return self._config

    @property
    def device(self) -> torch.device:
        """Return the device the backbone is currently on."""
        return next(self._backbone.parameters()).device


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name!r}; expected one of {sorted(mapping)}.")
    return mapping[name]
