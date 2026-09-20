from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from text_analysis_lab import dictionaries
from text_analysis_lab.dictionaries.providers import nltk as nltk_provider
from text_analysis_lab.translators import DictionaryTranslator


def test_nltk_opinion_lexicon_wrapper(monkeypatch) -> None:
    monkeypatch.setattr(
        nltk_provider,
        "_read_opinion_lexicon",
        lambda: (["good", "great"], ["bad", "awful"]),
    )
    dictionary = dictionaries.hu_liu()
    assert isinstance(dictionary, dictionaries.PolarityDictionary)
    assert dictionary.dictionary.entries["positive"] == ("good", "great")
    assert dictionary.dictionary.entries["negative"] == ("bad", "awful")
    assert dictionary.provenance is not None
    assert dictionary.provenance.provider == "NLTK"
    assert dictionary.provenance.resource == "opinion_lexicon"


def test_nltk_vader_wrapper_is_lexicon_only_valence(monkeypatch) -> None:
    monkeypatch.setattr(
        nltk_provider,
        "_read_vader_lexicon",
        lambda: {"good": 1.9, "bad": -2.5, ":)": 2.0},
    )
    dictionary = dictionaries.vader()
    assert isinstance(dictionary, dictionaries.ValenceDictionary)
    assert dictionary.values["valence"]["good"] == pytest.approx(1.9)
    assert dictionary.provenance is not None
    assert dictionary.provenance.license == "MIT License"
    assert "Lexicon-only" in (dictionary.provenance.notes or "")


def test_nltk_sentiwordnet_wrapper_preserves_three_synset_dimensions(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        nltk_provider,
        "_read_sentiwordnet",
        lambda: {
            "positive": {"happy.a.01": 0.875},
            "negative": {"happy.a.01": 0.0},
            "objective": {"happy.a.01": 0.125},
        },
    )
    dictionary = dictionaries.sentiwordnet()
    assert dictionary.dimensions == ("positive", "negative", "objective")
    assert dictionary.values["positive"]["happy.a.01"] == pytest.approx(0.875)
    assert dictionary.case_sensitive is True
    assert dictionary.provenance is not None
    assert dictionary.provenance.version == "3.0"


def test_nltk_wrapper_missing_resource_has_actionable_error(monkeypatch) -> None:
    def missing():
        raise LookupError("missing")

    monkeypatch.setattr(nltk_provider, "_read_vader_lexicon", missing)
    with pytest.raises(dictionaries.DictionaryResourceError) as excinfo:
        dictionaries.vader()
    message = str(excinfo.value)
    assert "vader_lexicon" in message
    assert "nltk.downloader" in message
    assert "download_nltk" in message


def test_afinn_adapter_uses_installed_provider_without_bundling(monkeypatch) -> None:
    module = types.ModuleType("afinn")

    class FakeAfinn:
        def __init__(self, language="en", emoticons=False):
            assert language == "en"
            assert emoticons is True
            self._dict = {"good": 3, "bad": -3, ":)": 2}

    module.Afinn = FakeAfinn
    monkeypatch.setitem(sys.modules, "afinn", module)

    dictionary = dictionaries.afinn(language="en", emoticons=True)
    assert dictionary.values["valence"] == {"good": 3.0, "bad": -3.0, ":)": 2.0}
    assert dictionary.provenance is not None
    assert dictionary.provenance.provider == "afinn"
    assert "ODbL" in (dictionary.provenance.license or "")


def test_external_dictionary_operator_uses_reference_and_hash_by_default(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        nltk_provider,
        "_read_vader_lexicon",
        lambda: {"good": 2.0, "bad": -2.0},
    )
    original = DictionaryTranslator(dictionaries.vader())
    state = original.to_json_state()
    assert state["dictionary"]["storage"] == "external"
    assert "good" not in json.dumps(state)
    operator_dir = tmp_path / "operator"
    original.save_to_dir(operator_dir, operator_id="op_external")
    assert not (operator_dir / "assets" / "dictionary.json").exists()

    restored = DictionaryTranslator.load_from_dir(operator_dir)
    assert restored.dictionary is not None
    assert restored.dictionary.provenance == original.dictionary.provenance
    assert restored.dictionary.name == "NLTK VADER Lexicon"


def test_external_dictionary_can_be_saved_for_offline_operator_reuse(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        nltk_provider,
        "_read_vader_lexicon",
        lambda: {"good": 2.0, "bad": -2.0},
    )
    original = DictionaryTranslator(dictionaries.vader(), save_dictionary=True)
    operator_dir = tmp_path / "operator"
    original.save_to_dir(operator_dir, operator_id="op_saved")
    assert (operator_dir / "assets" / "dictionary.json").exists()

    def unavailable():
        raise AssertionError("provider should not be consulted when local asset exists")

    monkeypatch.setattr(nltk_provider, "_read_vader_lexicon", unavailable)
    restored = DictionaryTranslator.load_from_dir(operator_dir)
    assert restored.dictionary is not None
    assert restored.dictionary.values["valence"]["good"] == pytest.approx(2.0)


def test_external_dictionary_hash_rejects_changed_provider(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        nltk_provider,
        "_read_vader_lexicon",
        lambda: {"good": 2.0, "bad": -2.0},
    )
    original = DictionaryTranslator(dictionaries.vader())
    operator_dir = tmp_path / "operator"
    original.save_to_dir(operator_dir, operator_id="op_hash")

    monkeypatch.setattr(
        nltk_provider,
        "_read_vader_lexicon",
        lambda: {"good": 9.0, "bad": -2.0},
    )
    with pytest.raises(Exception, match="content hash"):
        DictionaryTranslator.load_from_dir(operator_dir)


def test_dictionary_catalog_lists_supported_external_providers() -> None:
    catalog = dictionaries.catalog()
    assert {"hu_liu", "vader", "sentiwordnet", "afinn"}.issubset(set(catalog["name"]))
    assert set(catalog.columns) == {
        "name",
        "provider",
        "resource",
        "kind",
        "features",
        "license",
        "install",
    }
