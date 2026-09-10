from __future__ import annotations

import pytest

from text_analysis_lab import dictionaries


def _load_or_skip(loader, resource: str):
    try:
        return loader()
    except dictionaries.DictionaryResourceError:
        pytest.skip(
            f"NLTK resource {resource} is not installed; run "
            "`python -m nltk.downloader opinion_lexicon vader_lexicon "
            "sentiwordnet wordnet`."
        )


def test_real_nltk_opinion_lexicon_wrapper() -> None:
    dictionary = _load_or_skip(dictionaries.hu_liu, "opinion_lexicon")
    assert len(dictionary.dictionary.entries["positive"]) > 1000
    assert len(dictionary.dictionary.entries["negative"]) > 1000
    assert dictionary.provenance.resource == "opinion_lexicon"


def test_real_nltk_vader_wrapper() -> None:
    dictionary = _load_or_skip(dictionaries.vader, "vader_lexicon")
    assert len(dictionary.values["valence"]) > 1000
    assert dictionary.values["valence"]["good"] > 0
    assert dictionary.values["valence"]["bad"] < 0


def test_real_nltk_sentiwordnet_wrapper() -> None:
    dictionary = _load_or_skip(dictionaries.sentiwordnet, "sentiwordnet + wordnet")
    assert dictionary.dimensions == ("positive", "negative", "objective")
    assert len(dictionary.values["positive"]) > 100000
    assert dictionary.values["positive"]["happy.a.01"] > 0
    total = sum(dictionary.values[d]["happy.a.01"] for d in dictionary.dimensions)
    assert total == pytest.approx(1.0)


def test_real_afinn_wrapper_if_provider_installed() -> None:
    try:
        dictionary = dictionaries.afinn()
    except dictionaries.DictionaryResourceError:
        pytest.skip("Optional afinn provider is not installed.")
    assert len(dictionary.values["valence"]) > 1000
    assert dictionary.values["valence"]["good"] > 0
    assert dictionary.values["valence"]["bad"] < 0
