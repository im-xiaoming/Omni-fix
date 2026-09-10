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
    prompts = [_video_prompt(processor, config.media_instruction)] * len(paths)
    inputs = processor(
        text=prompts,
        videos=frames,
        size=_video_size(config),
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
    prompts = [_audio_prompt(processor, config.media_instruction)] * len(paths)
    inputs = processor(
        text=prompts,
        audio=waveforms,
        sampling_rate=config.audio_sample_rate,
        padding=True,
        return_tensors="pt",
    )
    inputs = _apply_beats_audio_slots(inputs, processor, backbone, config).to(_device(backbone))
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
    # The filler goes *before* the placeholder on this path only. Everywhere else
    # it follows the media, as training does, but the backbone's own M-RoPE index
    # cannot place trailing text once audio is interleaved into the video stream:
    # get_rope_index over-counts the interleaved chunks, skips its trailing-text
    # branch, and dies on a shape mismatch exactly the length of the filler.
    # Leading text goes through the same function without complaint.
    prompts = [config.media_instruction + _video_prompt(processor)] * len(paths)
    inputs = processor(
        text=prompts,
        videos=frames,
        audio=waveforms,
        sampling_rate=config.audio_sample_rate,
        size=_video_size(config),
        use_audio_in_video=True,
        padding=True,
        return_tensors="pt",
    )
    inputs = _apply_beats_audio_slots(inputs, processor, backbone, config).to(_device(backbone))
    return _forward_and_normalise(
        backbone,
        inputs,
        config,
        use_audio_in_video=True,
        input_raw_wav=_raw_wav(waveforms, backbone),
    )


# --------------------------------------------------------------------------- #
# Dual-modal encoders (added downstream, not part of the original release)     #
# --------------------------------------------------------------------------- #
#
# NOTE: everything from here to the "Internals" banner was written for this
# checkout; the released OmniRetriever code ships only the four encoders above.
# Without them the `tv` and `at` query sides are missing, so four of the twelve
# benchmark directions (a2tv, tv2a, v2at, at2v) cannot be scored at all.
#
# The prompt layout follows the training-time recipe in
# ``training/qwenvl/data/data_qwen.py::_prepare_submodal_input``: one media
# placeholder first, the caption after it, and no separate audio placeholder
# whenever a video is present. What it does *not* copy is the generic
# "Please describe the video." filler that upstream substitutes when a
# combination carries no caption, since both combinations here always do.
#
# Caveat worth knowing before trusting the numbers: upstream builds that prompt
# through ``apply_chat_template`` plus ``replace_multimodal_special_tokens``,
# while the released ``encode_*`` helpers hand the processor bare placeholder
# tokens. These two encoders follow the released inference convention so their
# output shares a space with the other four, not the training convention.


def encode_tv(backbone, processor, video_path, text, config):
    """Encode video and text jointly into the shared embedding space.

    Args:
        video_path: one path, or a sequence of paths, to video files.
        text: the matching caption, or a sequence of captions of equal length.

    Returns:
        Tensor of shape ``(N, D)``.
    """
    paths, texts = _pair(video_path, text, "video_path", "text")
    frames = [load_video_frames(p, num_frames=config.video_max_frames,
                                resolution=config.video_resolution) for p in paths]
    prompts = [_video_prompt(processor, t) for t in texts]
    inputs = processor(
        text=prompts,
        videos=frames,
        size=_video_size(config),
        padding=True,
        return_tensors="pt",
    ).to(_device(backbone))
    return _forward_and_normalise(backbone, inputs, config)


def encode_at(backbone, processor, audio_path, text, config):
    """Encode audio and text jointly into the shared embedding space.

    Args:
        audio_path: one path, or a sequence of paths, to audio files.
        text: the matching caption, or a sequence of captions of equal length.

    Returns:
        Tensor of shape ``(N, D)``.
    """
    paths, texts = _pair(audio_path, text, "audio_path", "text")
    waveforms = [load_audio_waveform(p,
                                     duration_sec=config.audio_duration_sec,
                                     sample_rate=config.audio_sample_rate) for p in paths]
    prompts = [_audio_prompt(processor, t) for t in texts]
    inputs = processor(
        text=prompts,
        audio=waveforms,
        sampling_rate=config.audio_sample_rate,
        padding=True,
        return_tensors="pt",
    )
    inputs = _apply_beats_audio_slots(inputs, processor, backbone, config).to(_device(backbone))
    return _forward_and_normalise(
        backbone, inputs, config, input_raw_wav=_raw_wav(waveforms, backbone)
    )


def _pair(media, text, media_name: str, text_name: str) -> tuple[list, list]:
    """Normalise a (media, text) argument pair into two equal-length lists."""
    media_list, text_list = _as_list(media), _as_list(text)
    if len(media_list) != len(text_list):
        raise ValueError(
            f"{media_name} and {text_name} must have the same length; "
            f"got {len(media_list)} and {len(text_list)}."
        )
    return media_list, text_list


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


# Filler the training pipeline appends whenever a modality combination carries
# no caption of its own; see
# ``training/qwenvl/data/data_qwen.py::_prepare_submodal_input``. ADDED to the
# inference path downstream: the released ``encode_*`` helpers passed the bare
# placeholder with no text at all, which is not a shape the backbone ever saw in
# training. Table S2 of the paper counts it too, describing the joint AV input as
# "video + audio + prompt".
MEDIA_INSTRUCTION = "Please describe the video."


def _video_prompt(processor, suffix: str = "") -> str:
    """Video placeholder, followed by ``suffix`` (a caption or the filler)."""
    return (
        processor.vision_bos_token + processor.video_token + processor.vision_eos_token + suffix
    )


def _audio_prompt(processor, suffix: str = "") -> str:
    """Audio placeholder, followed by ``suffix`` (a caption or the filler)."""
    return processor.audio_bos_token + processor.audio_token + processor.audio_eos_token + suffix


def _video_size(config) -> dict:
    """Pixel budget that keeps a frame at ``config.video_resolution``.

    ADDED downstream. Left to itself the video processor rescales a 224 px frame
    up to 336 px, which turns the 8-frame clip into 576 language-model tokens
    instead of 256. Table S2 of the paper puts a video-only forward at about 268
    tokens, i.e. 256 visual tokens plus the placeholder pair and the filler
    prompt, so the default rescale more than doubles the visual budget the model
    was trained on. ``max_pixels`` is ignored by this processor version; only a
    ``size`` dict takes effect.
    """
    edge = config.video_resolution
    return {"shortest_edge": _MIN_PIXELS, "longest_edge": edge * edge}


_MIN_PIXELS = 3136


def _apply_beats_audio_slots(inputs, processor, backbone, config):
    """Give every audio frame the second token slot the BEATs branch needs.

    ADDED downstream; the released code does not do this. With ``use_beats`` on
    and ``beats_only`` off, the WAVE forward pass interleaves one BEATs vector
    after every whisper vector, so the feature block it scatters into the token
    stream is twice as long as the audio placeholder the processor expands. The
    scatter is a ``masked_scatter``, which consumes as many rows as there are
    slots and silently drops the rest, so half the audio timeline never reaches
    the model. Upstream avoids this by doubling the placeholder after expansion
    (``training/qwenvl/data/data_qwen.py::_prepare_submodal_input``); this is the
    same edit applied to the already-tokenised ids.

    Set ``InferenceConfig.duplicate_audio_tokens = False`` to reproduce the
    released behaviour.
    """
    if not config.duplicate_audio_tokens or not _beats_interleaves(backbone):
        return inputs

    audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    ids_rows, mask_rows = [], []
    for ids, mask in zip(inputs["input_ids"], inputs["attention_mask"]):
        repeats = torch.where(ids == audio_token_id, 2, 1)
        ids_rows.append(torch.repeat_interleave(ids, repeats))
        mask_rows.append(torch.repeat_interleave(mask, repeats))

    # The processor left-pads the text stream and the fusion head pools the
    # final position, so the re-padding has to stay on the left.
    width = max(row.numel() for row in ids_rows)
    inputs["input_ids"] = torch.stack(
        [F.pad(row, (width - row.numel(), 0), value=pad_token_id) for row in ids_rows]
    )
    inputs["attention_mask"] = torch.stack(
        [F.pad(row, (width - row.numel(), 0), value=0) for row in mask_rows]
    )
    return inputs


def _beats_interleaves(backbone) -> bool:
    """True when the forward pass emits two feature rows per audio frame."""
    from omniretriever.models.beats_adaptor import find_beats_host

    try:
        host = find_beats_host(backbone)
    except AttributeError:
        return False
    return bool(getattr(host, "use_beats", False)) and not bool(getattr(host, "beats_only", False))


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
