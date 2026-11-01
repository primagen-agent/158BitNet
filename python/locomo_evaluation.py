"""Pinned LoCoMo dataset metadata and external scoring; evaluation only."""

from collections import Counter
import json
import re
import string
from nltk.stem import PorterStemmer

UPSTREAM = "https://github.com/snap-research/locomo"
UPSTREAM_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
DATA_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
STEMMER = PorterStemmer()


def normalize(text):
    text = str(text).lower().replace(",", "")
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the|and)\b", " ", text).split())


def token_f1(prediction, gold):
    prediction = [STEMMER.stem(w) for w in normalize(prediction).split()]
    gold = [STEMMER.stem(w) for w in normalize(gold).split()]
    common = sum((Counter(prediction) & Counter(gold)).values())
    return 2.0 * common / (len(prediction) + len(gold)) if common else 0.0


def official_f1(prediction, gold, category):
    """Category 1–4 scoring, including stemming and multi-answer means."""
    gold = str(gold)
    if category == 3:
        gold = gold.split(";")[0].strip()
    if category == 1:
        values = [
            max(token_f1(p.strip(), g.strip()) for p in prediction.split(","))
            for g in gold.split(",")
        ]
        return sum(values) / len(values)
    if category not in (2, 3, 4):
        raise ValueError("category 5 measures rejection, not answer F1")
    return token_f1(prediction, gold)


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with path.open("a") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")
        output.flush()
