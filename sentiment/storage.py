"""Tweet storage on any SQLAlchemy database: SQLite by default, MySQL via URL.

    DATABASE_URL=sqlite:///tweets.db                       (default)
    DATABASE_URL=mysql+mysqlconnector://user:pw@host/db    (pip install mysql-connector-python)

Credentials come from the environment, never from a committed file.
"""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (Column, DateTime, Float, MetaData, String, Table, Text, create_engine,
                        insert, select, update)
from sqlalchemy.engine import Engine

metadata = MetaData()

tweets = Table(
    "tweets", metadata,
    Column("id", String(64), primary_key=True),
    Column("text", Text, nullable=False),
    Column("created_at", DateTime),
    Column("sentiment", String(16)),
    Column("confidence", Float),
)

EXPORT_COLUMNS = ["id", "created_at", "text", "sentiment", "confidence"]


def get_engine(url: str | None = None) -> Engine:
    engine = create_engine(url or os.environ.get("DATABASE_URL", "sqlite:///tweets.db"))
    metadata.create_all(engine)
    return engine


def add_tweets(engine: Engine, rows: list[dict]) -> int:
    """Insert tweets ({id, text, created_at}); ids already stored are skipped."""
    if not rows:
        return 0
    ids = [r["id"] for r in rows]
    with engine.begin() as conn:
        existing = set(conn.execute(select(tweets.c.id).where(tweets.c.id.in_(ids))).scalars())
        new = [{"id": r["id"], "text": r["text"], "created_at": _as_datetime(r.get("created_at"))}
               for r in rows if r["id"] not in existing]
        if new:
            conn.execute(insert(tweets), new)
    return len(new)


def unclassified(engine: Engine) -> list[tuple[str, str]]:
    with engine.connect() as conn:
        return list(conn.execute(select(tweets.c.id, tweets.c.text).where(tweets.c.sentiment.is_(None))))


def set_sentiments(engine: Engine, results: list[tuple[str, str, float]]) -> None:
    with engine.begin() as conn:
        for tweet_id, label, conf in results:
            conn.execute(update(tweets).where(tweets.c.id == tweet_id).values(sentiment=label, confidence=conf))


def export_rows(engine: Engine) -> list[dict]:
    """Rows with explicit, named columns (the old export relied on SELECT * order)."""
    cols = [tweets.c[name] for name in EXPORT_COLUMNS]
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(select(*cols).order_by(tweets.c.created_at))]


def _as_datetime(value) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
