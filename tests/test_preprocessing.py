from sentiment.preprocessing import preprocess_text


def test_negations_are_kept():
    # The original pipeline removed NLTK stopwords, which include "not" and "don't".
    assert preprocess_text("This is NOT good") == "this is not good"
    assert "don't" in preprocess_text("I don't like it")


def test_urls_and_mentions_become_placeholders():
    out = preprocess_text("@bob check https://t.co/xyz now")
    assert out == "<user> check <url> now"


def test_hashtag_keeps_its_word():
    assert preprocess_text("Loving it #happy") == "loving it happy"


def test_elongation_is_capped_not_removed():
    assert preprocess_text("sooooo goooood") == "soo good"


def test_html_entities_and_emoji():
    assert preprocess_text("fish &amp; chips 😍") == "fish & chips 😍"


def test_empty_and_none():
    assert preprocess_text("") == ""
    assert preprocess_text(None) == ""
