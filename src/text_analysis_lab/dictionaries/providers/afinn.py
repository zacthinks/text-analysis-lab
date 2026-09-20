"""Optional adapter for the external ``afinn`` Python package."""

from __future__ import annotations

from text_analysis_lab.dictionaries.provenance import DictionaryProvenance
from text_analysis_lab.dictionaries.providers.nltk import DictionaryResourceError
from text_analysis_lab.dictionaries.source import DictionarySource
from text_analysis_lab.dictionaries.valence import ValenceDictionary


def afinn(*, language: str = "en", emoticons: bool = False) -> ValenceDictionary:
    """Load the lexicon supplied by an installed ``afinn`` package.

    TeAL does not redistribute AFINN data. Install the provider package
    separately (or use TeAL's ``lexicons`` extra), then this adapter freezes the
    resulting word/phrase scores into an ordinary :class:`ValenceDictionary`.
    """

    try:
        from afinn import Afinn
    except ImportError as exc:
        raise DictionaryResourceError(
            "The optional 'afinn' provider is not installed. Install with "
            "`uv add afinn` / `pip install afinn`, or install TeAL's `lexicons` extra."
        ) from exc

    analyzer = Afinn(language=str(language), emoticons=bool(emoticons))
    raw = getattr(analyzer, "_dict", None)
    if not isinstance(raw, dict):  # pragma: no cover - upstream API drift
        raise DictionaryResourceError(
            "The installed afinn package no longer exposes its loaded lexicon in the "
            "expected form; TeAL's adapter needs updating."
        )
    values = {str(term): float(score) for term, score in raw.items()}
    return ValenceDictionary(
        values,
        valuetype="fixed",
        case_sensitive=False,
        name=f"AFINN ({language})",
        provenance=DictionaryProvenance(
            provider="afinn",
            resource=f"AFINN-{language}",
            citation=(
                "Finn Arup Nielsen. 2011. A new ANEW: Evaluation of a word list "
                "for sentiment analysis in microblogs. arXiv:1103.2903."
            ),
            license=(
                "AFINN lexicon: Open Database License (ODbL) 1.0; provider code "
                "license is governed by the installed afinn package."
            ),
            source="https://github.com/fnielsen/afinn",
            notes=(
                "Loaded from the user's installed afinn package; TeAL does not bundle "
                "the AFINN word lists."
            ),
        ),
        source=DictionarySource(
            provider="afinn",
            resource="afinn",
            parameters={"language": str(language), "emoticons": bool(emoticons)},
        ),
    )
