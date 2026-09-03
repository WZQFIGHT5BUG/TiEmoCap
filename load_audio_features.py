"""Load audio features from the ``features/`` directory.

The feature files follow the same format as TiCaption's precomputed features::

    features/{sample_name}.pt -> {"audio": (T_a, 1024) float16, "text": (T_t, 1024) float16}

This module exposes only the audio (WavLM-large ``last_hidden_state``) branch.
Loading and validation mirror TiCaption's ``PrecomputedEmoCaptionDataset._load_features``
and ``collate_fn_precomputed``.

Example:
    from load_audio_features import load_audio, load_audio_batch

    audio = load_audio("2079_000023_angry")                 # (T_a, 1024) float16
    audio = load_audio("2079_000023_angry", to_float=True)  # (T_a, 1024) float32
    audio, mask = load_audio_batch(["2079_000023_angry", "1023_000038_neutral"])

Command line:
    python load_audio_features.py --names 2079_000023_angry 1023_000038_neutral
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

__all__ = ["load_audio", "load_audio_batch"]

# The ``features/`` directory is expected to sit next to this file.
FEATURES_DIR: Path = Path(__file__).resolve().parent / "features"
# Hidden size of the WavLM-large / CINO-large-v2 ``last_hidden_state``.
HIDDEN_DIM: int = 1024
# Optional suffixes accepted on a sample name.
_SUFFIXES: Tuple[str, ...] = (".pt", ".wav")


def _resolve_stem(name: str) -> str:
    """Strip an optional ``.pt`` / ``.wav`` suffix to get the sample stem."""
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _load_file(stem: str) -> dict:
    """Load the raw feature dict for a sample stem."""
    path = FEATURES_DIR / f"{stem}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # torch < 2.6 does not expose the ``weights_only`` argument.
        return torch.load(path, map_location="cpu")


def _check_feature(feature: torch.Tensor, stem: str, field: str) -> None:
    """Validate a feature tensor (mirrors TiCaption's ``_load_features``).

    Enforces a 2-D shape with ``HIDDEN_DIM`` columns, finite values, and
    non-zero content.
    """
    if not torch.is_tensor(feature) or feature.ndim != 2 or feature.shape[1] != HIDDEN_DIM:
        raise ValueError(
            f"Malformed {field} feature in {stem}.pt: "
            f"shape={getattr(feature, 'shape', type(feature))}"
        )
    if not torch.isfinite(feature).all():
        raise ValueError(f"{field} feature in {stem}.pt contains NaN/Inf")
    if torch.count_nonzero(feature) == 0:
        raise ValueError(f"{field} feature in {stem}.pt is all zeros")


def _pad(feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length features into a dense ``(B, T_max, HIDDEN_DIM)`` tensor."""
    batch_size = len(feats)
    max_len = max(feat.shape[0] for feat in feats)
    dim = feats[0].shape[1]
    padded = torch.zeros(batch_size, max_len, dim)  # float32, as in collate_fn_precomputed
    mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, feat in enumerate(feats):
        length = feat.shape[0]
        padded[i, :length] = feat
        mask[i, :length] = 1
    return padded, mask


def load_audio(name: str, to_float: bool = False) -> torch.Tensor:
    """Load the audio feature of a single sample.

    Args:
        name: Sample name, with or without the ``.pt`` / ``.wav`` suffix.
        to_float: If True, cast to ``float32``; otherwise keep the stored
            ``float16`` (matching TiCaption's ``_load_features``).

    Returns:
        Audio feature tensor of shape ``(T_a, 1024)``, where ``T_a`` is the
        variable number of audio frames.

    Raises:
        FileNotFoundError: If the feature file does not exist.
        ValueError: If the feature fails validation.
    """
    stem = _resolve_stem(name)
    data = _load_file(stem)
    audio = data.get("audio")
    _check_feature(audio, stem, "audio")
    return audio.float() if to_float else audio


def load_audio_batch(names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and pad a batch of audio features (mirrors ``collate_fn_precomputed``).

    Args:
        names: List of sample names.

    Returns:
        A tuple ``(audio, mask)``:
            - ``audio``: ``(B, T_max, 1024)`` float32, padded to the longest
              sequence in the batch.
            - ``mask``: ``(B, T_max)`` int64, 1 for valid frames, 0 for padding.
    """
    return _pad([load_audio(name) for name in names])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load audio features from features/.")
    parser.add_argument("--names", nargs="+", required=True, help="One or more sample names.")
    return parser


def main() -> None:
    """Command-line entry point: load and summarize audio features."""
    args = _build_parser().parse_args()
    audio, mask = load_audio_batch(args.names)
    print(f"audio: {tuple(audio.shape)} dtype={audio.dtype}")
    print(f"mask:  {tuple(mask.shape)} dtype={mask.dtype}")
    for name, row_mask in zip(args.names, mask):
        print(f"  {name}: valid frames={int(row_mask.sum())}")


if __name__ == "__main__":
    main()
