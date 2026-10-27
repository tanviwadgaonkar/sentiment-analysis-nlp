# Tweet Sentiment Analysis

[![CI](https://github.com/tanviwadgaonkar/sentiment-analysis-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/tanviwadgaonkar/sentiment-analysis-nlp/actions/workflows/ci.yml)

Classifies tweets as **negative / neutral / positive** with classical NLP (TF-IDF n-grams + linear models), trained and evaluated on the public **TweetEval** sentiment benchmark (SemEval-2017 Task 4A), and runs the model over your own tweets: import or collect them into a database, classify, export for Tableau.

```
                 ┌──────────── offline benchmark ────────────┐
TweetEval ──download (SHA-256 checked)──> train / val / test splits
                                               │
             preprocess -> word 1-2g + char 2-5g TF-IDF -> {NB, LogReg, LinearSVM}
                                               │  select on validation macro-recall
                                               ▼
                                 models/sentiment_model.joblib  (one sklearn Pipeline)
                                               │
 CSV / X API v2 ──import/collect──> tweets DB ──classify──> sentiment + confidence ──export──> CSV (Tableau)
 (SQLite by default, MySQL via DATABASE_URL)
```

## Results

TweetEval sentiment, official splits: 45,615 train / 2,000 validation / 12,284 test tweets. The model is chosen on **validation**; test is scored once. Macro-averaged recall is the official SemEval-2017 metric.

| Model | Val macro-recall | Test macro-recall | Test macro-F1 | Test accuracy |
|---|---|---|---|---|
| Naive Bayes, word TF-IDF (the original design) | 0.588 | 0.579 | 0.566 | 0.572 |
| **Logistic regression, word + char TF-IDF** ✅ | **0.692** | **0.629** | **0.620** | **0.623** |
| Linear SVM, word + char TF-IDF | 0.687 | 0.629 | 0.622 | 0.627 |

Per class (selected model, test): negative F1 0.624 · neutral F1 0.630 · positive F1 0.606.

Reproduce: `python -m sentiment download && python -m sentiment train` (≈2 minutes on a laptop CPU; CI re-runs it on every push). Numbers are from scikit-learn 1.9.1.

**Reading these numbers honestly.** This is a strong *classical* baseline, in line with the linear baselines reported in the TweetEval paper (Barbieri et al., 2020); fine-tuned transformers score substantially higher on the same split. The validation→test drop (0.69 → 0.63) is large; a likely cause (not measured here) is that the SemEval-2017 test tweets were collected later than the training data, so topics and vocabulary differ.

### Where it fails

From `python -m sentiment predict` with the trained model:

| Tweet | Predicted |
|---|---|
| I don't like this at all | negative (0.96) ✅ |
| best day ever!!! | positive (1.00) ✅ |
| this is not bad actually | **negative** (0.99) ❌ — "bad" outweighs the negation |
| the bus leaves at 5 | **negative** (0.66) ❌ — short factual tweets get little signal |

Bag-of-n-grams models can't compose meaning ("not" + "bad"), which is the main argument for a transformer upgrade.

## Known limitations and future work

- **No compositional meaning:** bag-of-n-grams can't model "not bad" (see the table above). The next step is a measured negation-scope feature, then a small fine-tuned transformer such as DistilRoBERTa, compared on the same validation split.
- **Domain shift:** trained on 2013–2017 SemEval tweets; slang, topics and emoji use drift, so expect lower accuracy on current posts.
- **Confidence isn't calibrated:** for the linear SVM it's a softmax over decision scores, and even logistic regression probabilities aren't checked for calibration. Treat it as a ranking signal.
- **Deployment:** this is a batch CLI (import → classify → export), not a web service; no hosted demo.

## Quick start

```bash
pip install -r requirements.txt

python -m sentiment download                 # TweetEval -> data/tweeteval/
python -m sentiment train                    # -> models/sentiment_model.joblib + metrics.json
python -m sentiment predict "I love this!" "worst day ever"

python -m sentiment import-csv data/sample_tweets.csv   # into sqlite:///tweets.db
python -m sentiment classify
python -m sentiment export tweets_with_sentiment.csv    # open in Tableau
```

Live collection (needs an X API plan with recent search): `TWITTER_BEARER_TOKEN=... python -m sentiment collect "coffee" --limit 200`. MySQL instead of SQLite: `pip install mysql-connector-python` and `DATABASE_URL=mysql+mysqlconnector://user:pw@host/db`. `data/sample_tweets.csv` holds five made-up example tweets for trying the flow.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q        # 19 tests, ~5 s, no downloads
```

Preprocessing rules (negations kept, URL/user placeholders, elongation), data loading and checksum verification, model selection and the saved pipeline on a tiny generated dataset, database import/dedupe/export, X API pagination with a fake client, and the full CLI flow.

## Design decisions

- **Negations are never removed.** The original pipeline stripped NLTK's English stopwords — which include *not*, *no*, *don't* — turning "not good" into "good". No words are removed now; TF-IDF already down-weights uninformative ones.
- **Word + character n-grams.** Word bigrams capture short phrases; character 2–5-grams (within words) are robust to misspellings, elongations ("sooo"), and hashtags, without NLTK stemmers or runtime corpus downloads.
- **One artifact.** Preprocessing, vectoriser and classifier are a single scikit-learn `Pipeline`, so a model can never be paired with the wrong vectoriser.
- **Labelled data first.** The original code trained on rows `WHERE sentiment IS NOT NULL`, but sentiment was only ever written by the classifier, so it could never train. The model now learns from a public labelled benchmark and is then applied to your unlabelled tweets.
- **Explicit export columns.** The old export named the columns of `SELECT *` positionally, in a different order from the table definition, mislabelling them.
- **Secrets from the environment.** Database URL and API token come from env vars, not a committed `config.json`.

## Layout

```
sentiment/
  preprocessing.py  tweet normalisation
  data.py           TweetEval download (checksummed) and loading
  model.py          candidate pipelines, validation-based selection, metrics
  storage.py        SQLAlchemy table: SQLite / MySQL
  collect.py        CSV import, X API v2 recent search
  __main__.py       CLI
tests/              pytest suite
data/sample_tweets.csv
```

## Data and licence

TweetEval is downloaded from its [repository](https://github.com/cardiffnlp/tweeteval) at run time and not redistributed here; see the TweetEval and SemEval-2017 Task 4 licences and X's terms. Citation: Barbieri, Camacho-Collados, Espinosa-Anke, Neves. *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.* Findings of EMNLP 2020.
