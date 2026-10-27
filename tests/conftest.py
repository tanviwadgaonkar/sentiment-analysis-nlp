"""A tiny dataset in TweetEval's file layout, so the full pipeline can be
tested in seconds without downloading the benchmark."""
import pytest

POS = ["love this so much", "what a great day", "this is amazing news", "so happy right now",
       "best concert ever", "really enjoyed it", "fantastic work team", "beautiful weather today"]
NEG = ["i hate this", "worst service ever", "this is terrible", "so angry right now",
       "not good at all", "really disappointed", "awful experience", "never buying again"]
NEU = ["the store opens at nine", "meeting moved to tuesday", "train leaves from platform two",
       "report due next week", "the game starts at eight", "new episode airs tonight",
       "office closed on monday", "update released for version two"]


def write_split(d, split, rows):
    (d / f"{split}_text.txt").write_text("\n".join(t for t, _ in rows) + "\n", encoding="utf-8")
    (d / f"{split}_labels.txt").write_text("\n".join(str(l) for _, l in rows) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("tweeteval")
    rows = [(t, 2) for t in POS] + [(t, 0) for t in NEG] + [(t, 1) for t in NEU]
    write_split(d, "train", rows * 3)
    write_split(d, "val", rows)
    write_split(d, "test", rows)
    return d


@pytest.fixture(scope="session")
def tiny_model(tiny_data, tmp_path_factory):
    from sentiment.model import train

    out = tmp_path_factory.mktemp("models")
    metrics = train(tiny_data, out)
    return out, metrics
