import json

import pytest

from sentiment import data
from sentiment.__main__ import classify_texts
from sentiment.model import load_model


def test_load_split_maps_label_ids_to_names(tiny_data):
    texts, labels = data.load_split(tiny_data, "val")
    assert len(texts) == len(labels) == 24
    assert set(labels) == {"negative", "neutral", "positive"}


def test_load_split_rejects_misaligned_files(tmp_path):
    (tmp_path / "val_text.txt").write_text("a\nb\n")
    (tmp_path / "val_labels.txt").write_text("0\n")
    with pytest.raises(ValueError, match="2 texts but 1 labels"):
        data.load_split(tmp_path, "val")


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    import io

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"tampered"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        data.download(tmp_path)
    assert not any(tmp_path.iterdir())


def test_training_selects_on_validation_and_saves_one_artifact(tiny_model):
    out, metrics = tiny_model
    assert metrics["selected_by"] == "validation macro_recall"
    best = metrics["best_model"]
    assert all(metrics["results"][best]["validation"]["macro_recall"] >= r["validation"]["macro_recall"]
               for r in metrics["results"].values())
    assert {p.name for p in out.iterdir()} == {"sentiment_model.joblib", "metrics.json"}
    assert json.loads((out / "metrics.json").read_text())["n"] == {"train": 72, "val": 24, "test": 24}


def test_saved_pipeline_classifies_raw_text(tiny_model):
    out, _ = tiny_model
    model = load_model(out)  # vectoriser is inside the pipeline: raw text in
    preds = classify_texts(model, ["what a great day", "worst service ever", "meeting moved to tuesday"])
    assert [p[0] for p in preds] == ["positive", "negative", "neutral"]
    assert all(0 < conf <= 1 for _, conf in preds)
