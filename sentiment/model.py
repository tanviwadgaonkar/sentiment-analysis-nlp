"""Train, select and evaluate sentiment classifiers.

    python -m sentiment train --data data/tweeteval --out models/

Each candidate is one scikit-learn Pipeline (preprocess -> TF-IDF -> model),
saved as a single artifact, so the vectoriser can never drift from the model
it was fitted with (the original code pickled them separately).

Selection uses the official *validation* split; the *test* split is scored
once, for the reported numbers. The headline metric is macro-averaged recall,
the official SemEval-2017 Task 4A metric, alongside accuracy and macro-F1.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from sentiment.data import LABELS, load_split
from sentiment.preprocessing import preprocess_text


def _features() -> FeatureUnion:
    # Word 1-2 grams capture phrases like "not good"; character 2-5 grams
    # within word boundaries handle misspellings, elongations and morphology.
    return FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True, token_pattern=r"<\w+>|[\w']+|[^\w\s]")),
        ("char", TfidfVectorizer(preprocessor=preprocess_text, analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=3, sublinear_tf=True)),
    ])


def candidates() -> dict[str, Pipeline]:
    word_only = lambda: TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2), min_df=2,  # noqa: E731
                                        sublinear_tf=True)
    return {
        # The original project's model (Naive Bayes on word TF-IDF), as the baseline.
        "naive_bayes_word_tfidf": Pipeline([("tfidf", word_only()), ("clf", ComplementNB())]),
        "logreg_word_char": Pipeline([("tfidf", _features()), ("clf", LogisticRegression(
            C=2.0, max_iter=2000, class_weight="balanced"))]),
        "linear_svm_word_char": Pipeline([("tfidf", _features()), ("clf", LinearSVC(
            C=0.2, class_weight="balanced"))]),
    }


def scores(y_true, y_pred) -> dict:
    return {
        "macro_recall": round(recall_score(y_true, y_pred, average="macro"), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }


def train(data_dir: str | Path, out_dir: str | Path = "models", models: dict | None = None) -> dict:
    X_train, y_train = load_split(data_dir, "train")
    X_val, y_val = load_split(data_dir, "val")
    X_test, y_test = load_split(data_dir, "test")

    results, fitted = {}, {}
    for name, pipe in (models or candidates()).items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = {"validation": scores(y_val, pipe.predict(X_val))}
        print(f"{name:24s} val macro-recall={results[name]['validation']['macro_recall']:.4f}")

    best = max(results, key=lambda n: results[n]["validation"]["macro_recall"])
    for name, pipe in fitted.items():  # test scores for the table; selection already done
        results[name]["test"] = scores(y_test, pipe.predict(X_test))
    report = classification_report(y_test, fitted[best].predict(X_test), labels=LABELS,
                                   output_dict=True, zero_division=0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], out / "sentiment_model.joblib", compress=3)
    metrics = {
        "best_model": best,
        "selected_by": "validation macro_recall",
        "dataset": "TweetEval sentiment (SemEval-2017 Task 4A)",
        "n": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "results": results,
        "best_test_per_class": {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score")}
                                for c in LABELS},
        "trained_with": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model(model_dir: str | Path = "models"):
    return joblib.load(Path(model_dir) / "sentiment_model.joblib")
