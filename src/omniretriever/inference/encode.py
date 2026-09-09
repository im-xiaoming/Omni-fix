"""Modality-specific encoding routines.

Each ``encode_*`` function takes a (single string or list-of-strings) input,
builds the appropriate multimodal prompt for WAVE-7B, runs a single forward
pass with the all-layer fusion head, and returns L2-normalised embeddings of
shape ``(N, D)`` with ``D = 3584``.

Inputs are batched automatically when a list is provided; users should chunk
manually if memory is tight.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from omniretriever.data.media import load_audio_waveform, load_video_frames

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #


def encode_text(backbone, processor, text, config):
    """Encode a string (or list of strings) into the shared embedding space."""
    texts = _as_list(text)
    inputs = processor(text=texts, padding=True, return_tensors="pt").to(_device(backbone))
    return _forward_and_normalise(backbone, inputs, config)


def encode_video(backbone, processor, video_path, config):
    """Encode video file(s) using the visual stream only."""
    paths = _as_list(video_path)
    frames = [load_video_frames(p, num_frames=config.video_max_frames,
                                resolution=config.video_resolution) for p in paths]
    prompts = [_video_prompt(processor)] * len(paths)
    inputs = processor(
        text=prompts,
        videos=frames,
        padding=True,
        return_tensors="pt",
    ).to(_device(backbone))
    return _forward_and_normalise(backbone, inputs, config)


def encode_audio(backbone, processor, audio_path, config):
    """Encode audio file(s) using the audio stream only."""
    paths = _as_list(audio_path)
    waveforms = [load_audio_waveform(p,
                                     duration_sec=config.audio_duration_sec,
                                     sample_rate=config.audio_sample_rate) for p in paths]
    prompts = [_audio_prompt(processor)] * len(paths)
    inputs = processor(
        text=prompts,
        audio=waveforms,
        sampling_rate=config.audio_sample_rate,
        padding=True,
        return_tensors="pt",
    ).to(_device(backbone))
    return _forward_and_normalise(
        backbone, inputs, config, input_raw_wav=_raw_wav(waveforms, backbone)
    )


def encode_av(backbone, processor, clip_path, config):
    """Encode multimodal clip(s) using both visual and audio streams."""
    paths = _as_list(clip_path)
    frames = [load_video_frames(p, num_frames=config.video_max_frames,
                                resolution=config.video_resolution) for p in paths]
    waveforms = [load_audio_waveform(p,
                                     duration_sec=config.audio_duration_sec,
                                     sample_rate=config.audio_sample_rate) for p in paths]
    # With use_audio_in_video the processor rewrites the plain video placeholder
    # into interleaved video/audio chunks itself, so the prompt stays the
    # video-only one and must not carry a separate audio placeholder.
    prompts = [_video_prompt(processor)] * len(paths)
    inputs = processor(
        text=prompts,
        videos=frames,
        audio=waveforms,
        sampling_rate=config.audio_sample_rate,
        use_audio_in_video=True,
        padding=True,
        return_tensors="pt",
    ).to(_device(backbone))
    return _forward_and_normalise(
        backbone,
        inputs,
        config,
        use_audio_in_video=True,
        input_raw_wav=_raw_wav(waveforms, backbone),
    )


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


def _video_prompt(processor) -> str:
    """Placeholder the processor expands into one token per video patch."""
    return processor.vision_bos_token + processor.video_token + processor.vision_eos_token


def _audio_prompt(processor) -> str:
    """Placeholder the processor expands into one token per audio frame."""
    return processor.audio_bos_token + processor.audio_token + processor.audio_eos_token


def _raw_wav(waveforms, backbone) -> list:
    """Waveforms for the BEATs branch, which reads raw audio rather than mel features.

    The remote code iterates ``input_raw_wav`` and feeds each entry to
    ``BEATs.extract_features``, so it must be a list of 1-D tensors, one per
    sample, at the model's audio sample rate. Leaving it out raises a TypeError
    inside the forward pass.
    """
    device = _device(backbone)
    return [torch.as_tensor(w, dtype=torch.float32).reshape(-1).to(device) for w in waveforms]


def _forward_and_normalise(backbone, inputs, config, **forward_kwargs) -> torch.Tensor:
    """Run a single forward pass and return (optionally L2-normalised) embeddings."""
    # ``pred_embeds=True`` is what makes the forward pass return embeddings and
    # stop there. Without it the WAVE-7B remote code falls through to its
    # contrastive training branch, which calls torch.distributed all_gather and
    # fails outside a launched process group.
    outputs = backbone(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        pred_embeds=True,
        **forward_kwargs,
    )
    # The all-layer fusion head concatenates the last-token hidden state of every
    # layer and projects it through ``classify_linear``. Upstream reads the
    # result off ``mllm_embeds``; ``text_embeds`` is only populated when label or
    # positive-pair inputs are supplied, which the encode_* helpers never do.
    embeddings = getattr(outputs, "mllm_embeds", None)
    if embeddings is None:
        raise RuntimeError(
            "Backbone did not return an 'mllm_embeds' field. Verify that the LoRA "
            "adapter is applied and that the WAVE-7B remote code is at the "
            "version expected by this release."
        )

    if config.normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    return embeddings


def _as_list(x) -> list:
    if isinstance(x, (str, Path)):
        return [str(x)]
    if isinstance(x, torch.Tensor):
        return [x]
    return [str(item) if isinstance(item, Path) else item for item in x]


def _device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device
