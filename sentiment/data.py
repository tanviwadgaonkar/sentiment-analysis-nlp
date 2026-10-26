"""Labelled data: the TweetEval sentiment benchmark (SemEval-2017 Task 4A).

    python -m sentiment download          # -> data/tweeteval/

Three classes (negative / neutral / positive) with the official train,
validation and test splits. Files are downloaded from the TweetEval
repository and checked against pinned SHA-256 hashes; they are not
redistributed here (see the TweetEval and SemEval licences).
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment"
LABELS = ["negative", "neutral", "positive"]
SPLITS = ("train", "val", "test")
SHA256 = {
    "train_text.txt": "368f01052ea6fd8ffc408a2a2e6ac9669e31542581a0396ef16591ea26eb98a6",
    "train_labels.txt": "122bfb1732fb6995b0e5c5f726c0ba457c469c3b6e60513007ce5037f23e65d4",
    "val_text.txt": "e5b021e6fc45064c260b09814b803d8f56cada519c4d952d72f43d48a350a964",
    "val_labels.txt": "b4566926c72e2e4e2916c864def94e76c4cdde52446af2c7ba4fc2006e057e51",
    "test_text.txt": "09a93a55c63fd93f97485ef7302889d7edb4091cd49733aa37da094f0bfa0675",
    "test_labels.txt": "6afb4afe9374d1f983bcf9a7c79b108d0f37fdf020a83f30488309bed215db9d",
}


def download(out_dir: str | Path = "data/tweeteval") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in SHA256.items():
        target = out / name
        if target.exists() and _sha256(target.read_bytes()) == expected:
            continue
        with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as resp:
            payload = resp.read()
        if _sha256(payload) != expected:
            raise RuntimeError(f"checksum mismatch for {name}; the upstream file changed")
        target.write_bytes(payload)
    return out


def load_split(data_dir: str | Path, split: str) -> tuple[list[str], list[str]]:
    """Return ``(texts, labels)`` for one split, labels as class names."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    d = Path(data_dir)
    texts = (d / f"{split}_text.txt").read_text(encoding="utf-8").splitlines()
    labels = [LABELS[int(x)] for x in (d / f"{split}_labels.txt").read_text(encoding="utf-8").split()]
    if len(texts) != len(labels):
        raise ValueError(f"{split}: {len(texts)} texts but {len(labels)} labels")
    return texts, labels


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
