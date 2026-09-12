"""Patch the WAVE-7B remote code so the AV path accepts a trailing token.

Run once against the downloaded model directory, before anything loads it::

    python scripts/patch_wave_rope.py /path/to/WAVE-7B

What it fixes
-------------
``get_rope_index`` walks the prompt and accumulates M-RoPE positions into
``llm_pos_ids_list``. After the media chunks it appends positions for whatever
text is left, guarded by a running offset ``st``::

    if st < len(input_tokens):
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(...)

With ``use_audio_in_video`` that offset over-counts: the interleaved audio and
video chunks are added to ``st`` as if they sat end to end, when the chunking
merged them. ``st`` then lands past the real end, the guard is false, and any
trailing token gets no position at all. The scatter that follows fails with a
shape mismatch exactly as long as the trailing text.

The consequence is not an edge case. Training closes every prompt with
``<|im_end|>`` and the fusion head pools a fixed final position, so the AV path
cannot be given the token the model was trained to write its embedding into.

The fix does not touch the arithmetic. It asks how many positions were actually
produced and fills the remainder, which is self-correcting whatever ``st`` says.
Prompts that already worked are untouched: there ``st`` and the produced count
agree, both guards are false, and nothing is appended.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = "modeling_qwen2_5_omni.py"

OLD = """                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
"""

NEW = """                # PATCHED: `st` over-counts once audio is interleaved into the
                # video stream, so it cannot say how much prompt is left. Count
                # the positions actually produced instead; the two agree on
                # every prompt that already worked.
                produced = sum(int(p.shape[-1]) for p in llm_pos_ids_list)
                if produced < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - produced
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
"""


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[0])
        print("usage: python scripts/patch_wave_rope.py /path/to/WAVE-7B")
        return 2

    path = Path(argv[0])
    if path.is_dir():
        path = path / TARGET
    if not path.is_file():
        print(f"not found: {path}")
        return 1

    source = path.read_text(encoding="utf-8")

    if "PATCHED: `st` over-counts" in source:
        print(f"already patched: {path}")
        return 0

    occurrences = source.count(OLD)
    if occurrences != 1:
        print(f"expected exactly one match in {path}, found {occurrences}; not patching")
        return 1

    backup = path.with_suffix(path.suffix + ".prepatch")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")

    path.write_text(source.replace(OLD, NEW), encoding="utf-8")
    print(f"patched {path}")
    print(f"original kept at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
