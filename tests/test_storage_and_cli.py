import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentiment import collect, storage
from sentiment.__main__ import main

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_tweets.csv"


@pytest.fixture
def engine(tmp_path):
    return storage.get_engine(f"sqlite:///{tmp_path / 't.db'}")


def test_sample_csv_is_well_formed():
    rows = collect.read_csv(SAMPLE)
    assert len(rows) == 5 and rows[0]["id"] == "1001"


def test_import_skips_duplicates(engine):
    rows = collect.read_csv(SAMPLE)
    assert storage.add_tweets(engine, rows) == 5
    assert storage.add_tweets(engine, rows) == 0


def test_csv_without_required_columns_is_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("tweet\nhello\n")
    with pytest.raises(ValueError, match="missing columns"):
        collect.read_csv(p)


def test_export_columns_are_named_not_positional(engine):
    storage.add_tweets(engine, [{"id": "1", "text": "hi", "created_at": "2023-06-01 10:00:00"}])
    storage.set_sentiments(engine, [("1", "neutral", 0.8)])
    [row] = storage.export_rows(engine)
    assert list(row) == storage.EXPORT_COLUMNS
    assert (row["text"], row["sentiment"], row["confidence"]) == ("hi", "neutral", 0.8)


def test_live_collection_paginates_with_fake_client():
    pages = [
        SimpleNamespace(data=[SimpleNamespace(id=i, text=f"t{i}", created_at=None) for i in range(10)],
                        meta={"next_token": "p2"}),
        SimpleNamespace(data=[SimpleNamespace(id=i, text=f"t{i}", created_at=None) for i in range(10, 15)],
                        meta={}),
    ]
    calls = []

    class FakeClient:
        def search_recent_tweets(self, **kw):
            calls.append(kw)
            return pages[len(calls) - 1]

    rows = collect.search_recent("coffee", limit=50, client=FakeClient())
    assert [r["id"] for r in rows] == [str(i) for i in range(15)]
    assert "-is:retweet" in calls[0]["query"] and calls[1]["next_token"] == "p2"


def test_live_collection_needs_a_token(monkeypatch):
    monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
    pytest.importorskip("tweepy")
    with pytest.raises(RuntimeError, match="TWITTER_BEARER_TOKEN"):
        collect.search_recent("coffee")


def test_cli_end_to_end(tiny_model, tmp_path, capsys):
    models, _ = tiny_model
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    out_csv = tmp_path / "export.csv"
    common = ["--db", db, "--models", str(models)]

    assert main(common + ["import-csv", str(SAMPLE)]) == 0
    assert main(common + ["classify"]) == 0
    assert main(common + ["classify"]) == 0  # nothing left to do the second time
    assert main(common + ["export", str(out_csv)]) == 0

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(rows) == 5
    assert all(r["sentiment"] in {"negative", "neutral", "positive"} for r in rows)
    assert "classified 5 tweets" in capsys.readouterr().out
