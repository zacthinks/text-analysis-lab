"""Discoverable catalog of TeAL dictionary-provider adapters."""

from __future__ import annotations

import pandas as pd

_ROWS = [
    {
        "name": "hu_liu",
        "provider": "NLTK",
        "resource": "opinion_lexicon",
        "kind": "polarity",
        "features": "words",
        "license": "CC BY 4.0 (NLTK data metadata)",
        "install": "python -m nltk.downloader opinion_lexicon",
    },
    {
        "name": "vader",
        "provider": "NLTK",
        "resource": "vader_lexicon",
        "kind": "valence",
        "features": "words/phrases",
        "license": "MIT",
        "install": "python -m nltk.downloader vader_lexicon",
    },
    {
        "name": "sentiwordnet",
        "provider": "NLTK",
        "resource": "sentiwordnet + wordnet",
        "kind": "multidimensional valence",
        "features": "WordNet synset names",
        "license": "CC BY-SA 3.0 (NLTK data metadata)",
        "install": "python -m nltk.downloader sentiwordnet wordnet",
    },
    {
        "name": "afinn",
        "provider": "afinn package",
        "resource": "AFINN",
        "kind": "valence",
        "features": "words/phrases",
        "license": "ODbL 1.0 (lexicon)",
        "install": "pip install afinn",
    },
]


def catalog() -> pd.DataFrame:
    """Return the standard external dictionary adapters known to TeAL."""

    return pd.DataFrame(_ROWS).copy()
