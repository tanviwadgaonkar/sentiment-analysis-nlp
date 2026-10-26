"""Tweet normalisation for a bag-of-n-grams model.

Design choices (each one matters for sentiment, not just for tidiness):

- **Negations are kept.** A generic English stopword list removes "not",
  "no", "never", "don't" — which turns "not good" into "good". The original
  pipeline did exactly that. We remove no words at all; TF-IDF already
  down-weights words that appear everywhere.
- **URLs and @mentions become placeholder tokens** (``<url>``, ``<user>``):
  the specific link or handle carries no sentiment, but its presence can.
- **Hashtags keep their word** (``#happy`` -> ``happy``).
- **Elongations are capped** (``sooooo`` -> ``soo``) so variants share a feature
  while still differing from the plain word.
- **Emoticons and emoji survive** — they are among the strongest signals.
- No stemming/lemmatisation: character n-grams in the model capture
  morphology without NLTK downloads at import time.
"""
from __future__ import annotations

import html
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_ELONGATED = re.compile(r"(\w)\1{2,}")
_WS = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    text = html.unescape(text or "")
    text = _URL.sub(" <url> ", text)
    text = _USER.sub(" <user> ", text)
    text = _HASHTAG.sub(r" \1 ", text)
    text = text.lower()
    text = _ELONGATED.sub(r"\1\1", text)
    return _WS.sub(" ", text).strip()
