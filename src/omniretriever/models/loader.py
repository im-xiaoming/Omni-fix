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
        duplicate_audio_tokens: whether to give each audio frame the second
            token slot the interleaved BEATs branch fills. Added downstream;
            set it to ``False`` to reproduce the released behaviour, which
            leaves half the audio feature block unscattered. See
            ``omniretriever.inference.encode._apply_beats_audio_slots``.
    """

    video_max_frames: int = DEFAULT_VIDEO_FRAMES
    video_resolution: int = DEFAULT_VIDEO_RESOLUTION
    audio_duration_sec: int = DEFAULT_AUDIO_DURATION_SEC
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    embed_dim: int = DEFAULT_EMBED_DIM
    normalize: bool = True
    precision: str = "bfloat16"
    duplicate_audio_tokens: bool = True

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
        duplicate_audio_tokens: bool = True,
    ) -> "OmniRetriever":
        """Load WAVE-7B and apply the OmniRetriever LoRA adapter.

        Args:
            base_model: path (or HF hub id) of the WAVE-7B backbone.
            adapter: path of the released LoRA adapter directory.
            device: PyTorch device string.
            dtype: precision (one of ``float32`` / ``bfloat16`` / ``float16``).
            config: optional :class:`InferenceConfig` override. When given, it
                is used as-is and ``duplicate_audio_tokens`` is ignored.
            duplicate_audio_tokens: see the field of the same name on
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
            precision=dtype, duplicate_audio_tokens=duplicate_audio_tokens
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
