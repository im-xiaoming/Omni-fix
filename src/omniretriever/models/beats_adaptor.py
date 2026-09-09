"""BEATs audio-adaptor wiring for the WAVE-7B backbone.

The WAVE-7B Hugging Face repository already ships the BEATs implementation
(``beats_BEATs.py``) and instantiates it from ``config.audio_config.beats_cfg``
inside ``Qwen2_5OmniThinkerForConditionalGeneration.__init__``. The fine-tuned
``beats.*``, ``beats_ln.*`` and ``beats_proj.*`` tensors are part of the WAVE-7B
shards, so ``from_pretrained`` normally restores them along with the rest of the
model.

This module therefore does two things:

* verify that the encoder really is present and populated, and
* fall back to a standalone BEATs ``.pt`` checkpoint when it is not.

Weights already present in the shards are never overwritten. That matches
upstream, where BEATs stays frozen for the whole training run: only
``classify_linear``, ``beats_ln`` and ``beats_proj`` are ever unfrozen (see
``qwenvl/train/train_qwen.py::set_model``), and those three are exactly the
``modules_to_save`` of the released LoRA adapter. The ``beats.*`` tensors in the
WAVE-7B shards are a bfloat16 cast of the stock ``BEATs_iter3_plus_AS2M.pt``, so
re-loading the standalone file would change nothing.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = ["attach_beats", "find_beats_host"]


def find_beats_host(model) -> nn.Module:
    """Return the submodule that owns (or should own) the BEATs encoder.

    ``AutoModel`` may hand back the thinker directly, or a wrapper such as the
    full Qwen2.5-Omni model or a PEFT-wrapped module. Walk down to the module
    that declares ``use_beats``.
    """
    for candidate in (
        model,
        getattr(model, "thinker", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ):
        if candidate is not None and hasattr(candidate, "use_beats"):
            return candidate

    for module in model.modules():
        if hasattr(module, "use_beats"):
            return module

    raise AttributeError(
        "No BEATs-capable submodule found on the backbone. The loaded model does "
        "not look like a WAVE-7B checkpoint (expected an attribute 'use_beats' "
        "on the thinker). Check that config.json contains audio_config.beats_cfg."
    )


def attach_beats(model, beats_ckpt_path: Path) -> None:
    """Ensure the BEATs encoder is attached, populated and frozen.

    Args:
        model: the loaded WAVE-7B backbone.
        beats_ckpt_path: standalone BEATs checkpoint, used only as a fallback.
    """
    host = find_beats_host(model)

    if not getattr(host, "use_beats", False):
        raise RuntimeError(
            "The backbone was loaded without BEATs support (use_beats is False). "
            "This happens when config.json has no audio_config.beats_cfg entry; "
            "audio-anchored retrieval will not work with this checkpoint."
        )

    if getattr(host, "beats", None) is None:
        logger.info("BEATs encoder absent from the backbone; building it from %s", beats_ckpt_path)
        _build_beats(host, model, beats_ckpt_path)
    elif _looks_uninitialised(host.beats):
        logger.warning(
            "BEATs encoder is present but its weights look uninitialised; "
            "loading them from %s",
            beats_ckpt_path,
        )
        _load_beats_state(host.beats, beats_ckpt_path)
    else:
        logger.info("BEATs encoder already restored from the WAVE-7B shards; keeping those weights")

    _check_projection_head(host)

    # BEATs itself is frozen, as it is upstream. The LayerNorm and projector that
    # map its hidden states into the LLM token stream are restored by the LoRA
    # adapter (modules_to_save = classify_linear, beats_ln, beats_proj), so they
    # are left alone here. Train/eval mode is left to the caller, matching
    # upstream; OmniRetriever.from_pretrained puts the whole backbone in eval.
    for param in host.beats.parameters():
        param.requires_grad_(False)


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #


def _beats_module(model):
    """Import the ``beats_BEATs`` module that ships with the WAVE-7B repo."""
    remote_module = type(model).__module__
    package = remote_module.rsplit(".", 1)[0]
    try:
        return importlib.import_module(package + ".beats_BEATs")
    except ImportError as exc:
        raise ImportError(
            "Could not import 'beats_BEATs' from the WAVE-7B remote code at "
            f"'{package}'. Make sure the model directory contains beats_BEATs.py "
            "and that the model was loaded with trust_remote_code=True."
        ) from exc


def _build_beats(host, model, beats_ckpt_path: Path) -> None:
    beats_mod = _beats_module(model)
    checkpoint = _read_checkpoint(beats_ckpt_path)

    cfg_dict = getattr(getattr(model.config, "audio_config", None), "beats_cfg", None)
    cfg_dict = cfg_dict or checkpoint.get("cfg")
    if cfg_dict is None:
        raise RuntimeError(
            f"No BEATs config found in the model config or in {beats_ckpt_path}."
        )

    encoder = beats_mod.BEATs(beats_mod.BEATsConfig(cfg_dict))
    _load_state_dict(encoder, checkpoint["model"], source=beats_ckpt_path)

    reference = next(model.parameters())
    host.beats = encoder.to(device=reference.device, dtype=reference.dtype)
    host.beats_avg_pooler = nn.AvgPool1d(2, stride=2)


def _load_beats_state(encoder: nn.Module, beats_ckpt_path: Path) -> None:
    checkpoint = _read_checkpoint(beats_ckpt_path)
    _load_state_dict(encoder, checkpoint["model"], source=beats_ckpt_path)


def _read_checkpoint(beats_ckpt_path: Path) -> dict:
    checkpoint = torch.load(str(beats_ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(
            f"{beats_ckpt_path} is not a BEATs checkpoint; expected a dict with a "
            "'model' key holding the encoder state dict."
        )
    return checkpoint


def _load_state_dict(encoder: nn.Module, state_dict: dict, *, source: Path) -> None:
    reference = next(encoder.parameters())
    state_dict = {k: v.to(reference.dtype) for k, v in state_dict.items()}
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)

    # The classification head of the AudioSet-finetuned checkpoint is unused here.
    missing = [k for k in missing if not k.startswith("predictor")]
    unexpected = [k for k in unexpected if not k.startswith("predictor")]
    if missing:
        raise RuntimeError(
            f"BEATs weights in {source} do not match the configured encoder; "
            f"{len(missing)} tensors are missing, e.g. {missing[:5]}."
        )
    if unexpected:
        logger.debug("Ignoring %d unexpected BEATs tensors, e.g. %s", len(unexpected), unexpected[:5])


def _looks_uninitialised(encoder: nn.Module) -> bool:
    """Heuristic check that ``from_pretrained`` actually restored the encoder.

    Transformers leaves modules whose tensors were absent from the checkpoint at
    their freshly initialised values. For the BEATs encoder that means an
    all-ones final LayerNorm weight, which a trained encoder never has.
    """
    layer_norm = getattr(getattr(encoder, "encoder", None), "layer_norm", None)
    if layer_norm is None or getattr(layer_norm, "weight", None) is None:
        return False
    return bool(torch.all(layer_norm.weight == 1.0))


def _check_projection_head(host) -> None:
    for name in ("beats_ln", "beats_proj", "beats_avg_pooler"):
        if getattr(host, name, None) is None:
            raise RuntimeError(
                f"The backbone is missing '{name}', which maps BEATs features into "
                "the LLM token stream. The WAVE-7B remote code normally creates it; "
                "check that modeling_qwen2_5_omni.py in the model directory is the "
                "WAVE variant and not stock Qwen2.5-Omni."
            )
