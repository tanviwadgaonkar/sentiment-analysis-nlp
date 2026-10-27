"""Command line: ``python -m sentiment <command>``.

    download                       fetch the TweetEval benchmark (checksummed)
    train                          train candidates, select on validation, score on test
    predict "some tweet" ...       classify text directly
    import-csv data/tweets.csv     load tweets into the database
    collect "query" --limit 200    live tweets via the X API v2 (needs TWITTER_BEARER_TOKEN)
    classify                       label every unclassified tweet in the database
    export out.csv                 CSV for Tableau / BI tools
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

from sentiment import collect, data, storage
from sentiment.model import load_model, train


def classify_texts(model, texts: list[str]) -> list[tuple[str, float]]:
    """(label, confidence) per text. Confidence is the top-class probability
    when the model provides it, else a softmax over decision scores."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
    else:
        scores = np.atleast_2d(model.decision_function(texts))
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
    classes = list(model.classes_)
    return [(classes[int(p.argmax())], round(float(p.max()), 4)) for p in probs]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sentiment", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", help="SQLAlchemy URL (default: $DATABASE_URL or sqlite:///tweets.db)")
    p.add_argument("--models", default="models")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").add_argument("--out", default="data/tweeteval")
    sub.add_parser("train").add_argument("--data", default="data/tweeteval")
    sub.add_parser("predict").add_argument("texts", nargs="+")
    sub.add_parser("import-csv").add_argument("path")
    c = sub.add_parser("collect")
    c.add_argument("query")
    c.add_argument("--limit", type=int, default=100)
    sub.add_parser("classify")
    sub.add_parser("export").add_argument("path")
    a = p.parse_args(argv)

    if a.cmd == "download":
        print(f"downloaded to {data.download(a.out)}")
    elif a.cmd == "train":
        m = train(a.data, a.models)
        print(json.dumps({"best_model": m["best_model"], "test": m["results"][m["best_model"]]["test"]}, indent=2))
    elif a.cmd == "predict":
        for text, (label, conf) in zip(a.texts, classify_texts(load_model(a.models), a.texts)):
            print(f"{label:8s} {conf:.2f}  {text}")
    elif a.cmd == "import-csv":
        print(f"added {storage.add_tweets(storage.get_engine(a.db), collect.read_csv(a.path))} tweets")
    elif a.cmd == "collect":
        rows = collect.search_recent(a.query, a.limit)
        print(f"added {storage.add_tweets(storage.get_engine(a.db), rows)} of {len(rows)} tweets")
    elif a.cmd == "classify":
        engine = storage.get_engine(a.db)
        pending = storage.unclassified(engine)
        if pending:
            labels = classify_texts(load_model(a.models), [t for _, t in pending])
            storage.set_sentiments(engine, [(i, lab, conf) for (i, _), (lab, conf) in zip(pending, labels)])
        print(f"classified {len(pending)} tweets")
    elif a.cmd == "export":
        rows = storage.export_rows(storage.get_engine(a.db))
        with open(a.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=storage.EXPORT_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {a.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
