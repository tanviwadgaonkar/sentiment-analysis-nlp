"""Get tweets into the database: from a CSV file, or live from the X/Twitter API.

The v1.1 ``api.search`` endpoint used originally no longer exists; live
collection uses API v2 recent search (``tweepy.Client``) with a bearer token
from ``TWITTER_BEARER_TOKEN``. Recent search requires a paid API tier, so CSV
import is the default path.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path


def read_csv(path: str | Path) -> list[dict]:
    """Rows with ``id`` and ``text`` columns (``created_at`` optional)."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"id", "text"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return [{"id": r["id"].strip(), "text": r["text"], "created_at": r.get("created_at") or None}
                for r in reader if r["id"].strip() and r["text"].strip()]


def search_recent(query: str, limit: int = 100, client=None) -> list[dict]:
    """Recent English tweets matching ``query`` (retweets excluded)."""
    if client is None:
        import tweepy

        token = os.environ.get("TWITTER_BEARER_TOKEN")
        if not token:
            raise RuntimeError("set TWITTER_BEARER_TOKEN to collect live tweets")
        client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)

    rows: list[dict] = []
    next_token = None
    while len(rows) < limit:
        resp = client.search_recent_tweets(
            query=f"({query}) lang:en -is:retweet", tweet_fields=["created_at"],
            max_results=max(10, min(100, limit - len(rows))), next_token=next_token,
        )
        rows += [{"id": str(t.id), "text": t.text, "created_at": t.created_at} for t in (resp.data or [])]
        next_token = (resp.meta or {}).get("next_token")
        if not next_token or not resp.data:
            break
    return rows[:limit]
