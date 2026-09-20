"""Adapters for sentiment/content-analysis lexicons distributed through NLTK."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from text_analysis_lab.dictionaries.dictionary import Dictionary
from text_analysis_lab.dictionaries.polarity import PolarityDictionary
from text_analysis_lab.dictionaries.provenance import DictionaryProvenance
from text_analysis_lab.dictionaries.source import DictionarySource
from text_analysis_lab.dictionaries.valence import ValenceDictionary

T = TypeVar("T")


class DictionaryResourceError(RuntimeError):
    """Raised when an external dictionary package/data resource is unavailable."""


_HU_LIU_CITATION = (
    'Minqing Hu and Bing Liu. 2004. "Mining and Summarizing Customer Reviews." '
    "Proceedings of KDD-04."
)
_VADER_CITATION = (
    "C.J. Hutto and E.E. Gilbert. 2014. VADER: A Parsimonious Rule-based Model "
    "for Sentiment Analysis of Social Media Text. ICWSM-14."
)
_SENTIWORDNET_CITATION = (
    "Stefano Baccianella, Andrea Esuli, and Fabrizio Sebastiani. 2010. "
    "SentiWordNet 3.0: An Enhanced Lexical Resource for Sentiment Analysis and "
    "Opinion Mining. LREC 2010."
)


def opinion_lexicon(*, download: bool = False) -> PolarityDictionary:
    """Load NLTK's Hu-Liu Opinion Lexicon as a polarity dictionary.

    The underlying NLTK data package is ``opinion_lexicon``. TeAL does not copy
    the lexicon into its own distribution; NLTK remains the data provider.
    """

    positive, negative = _load_with_nltk_data(
        ("opinion_lexicon",), _read_opinion_lexicon, download=download
    )
    base = Dictionary(
        {"positive": positive, "negative": negative},
        valuetype="fixed",
        case_sensitive=False,
        name="NLTK Opinion Lexicon (Hu-Liu)",
        provenance=DictionaryProvenance(
            provider="NLTK",
            resource="opinion_lexicon",
            citation=_HU_LIU_CITATION,
            license=(
                "CC BY 4.0 according to the current NLTK data license inventory; "
                "NLTK's corpus-reader documentation also notes distribution with permission."
            ),
            source="https://www.nltk.org/",
            notes="Positive/negative English opinion-word lists.",
        ),
        source=DictionarySource(provider="nltk", resource="opinion_lexicon"),
    )
    return PolarityDictionary(base, positive="positive", negative="negative")


def vader(*, download: bool = False) -> ValenceDictionary:
    """Load NLTK's VADER lexicon as a lexical valence dictionary.

    This exposes the lexicon's raw word/phrase valence ratings only. It does not
    run VADER's sentence-level negation, capitalization, punctuation, booster,
    or contrastive-conjunction rules.
    """

    values = _load_with_nltk_data(
        ("vader_lexicon",), _read_vader_lexicon, download=download
    )
    return ValenceDictionary(
        values,
        valuetype="fixed",
        case_sensitive=False,
        name="NLTK VADER Lexicon",
        provenance=DictionaryProvenance(
            provider="NLTK",
            resource="vader_lexicon",
            citation=_VADER_CITATION,
            license="MIT License",
            source="https://www.nltk.org/api/nltk.sentiment.vader.html",
            notes=(
                "Lexicon-only adapter. TeAL dictionary translation does not reproduce "
                "VADER's rule-based sentence sentiment analyzer."
            ),
        ),
        source=DictionarySource(provider="nltk", resource="vader"),
    )


def sentiwordnet(*, download: bool = False) -> ValenceDictionary:
    """Load NLTK SentiWordNet as a three-dimensional synset dictionary.

    Features are WordNet synset names such as ``happy.a.01`` rather than surface
    words. The returned dimensions are ``positive``, ``negative``, and
    ``objective``. This is therefore appropriate for a matrix whose features
    have already been sense-disambiguated to WordNet synsets.
    """

    values = _load_with_nltk_data(
        ("sentiwordnet", "wordnet"), _read_sentiwordnet, download=download
    )
    return ValenceDictionary(
        values,
        valuetype="fixed",
        case_sensitive=True,
        name="NLTK SentiWordNet 3.0",
        provenance=DictionaryProvenance(
            provider="NLTK",
            resource="sentiwordnet",
            version="3.0",
            citation=_SENTIWORDNET_CITATION,
            license="CC BY-SA 3.0 according to the current NLTK data metadata.",
            source="https://www.nltk.org/howto/sentiwordnet.html",
            notes=(
                "Synset-level resource. Requires both NLTK sentiwordnet and wordnet data."
            ),
        ),
        source=DictionarySource(provider="nltk", resource="sentiwordnet"),
    )


def download(resources: str | Iterable[str] = "all", *, quiet: bool = False) -> None:
    """Explicitly download TeAL-supported NLTK dictionary data packages."""

    import nltk

    aliases = {
        "opinion_lexicon": ("opinion_lexicon",),
        "hu_liu": ("opinion_lexicon",),
        "vader": ("vader_lexicon",),
        "vader_lexicon": ("vader_lexicon",),
        "sentiwordnet": ("sentiwordnet", "wordnet"),
        "wordnet": ("wordnet",),
        "all": ("opinion_lexicon", "vader_lexicon", "sentiwordnet", "wordnet"),
    }
    requested = [resources] if isinstance(resources, str) else list(resources)
    packages: list[str] = []
    for value in requested:
        key = str(value).lower()
        if key not in aliases:
            raise ValueError(
                f"Unknown NLTK dictionary resource {value!r}; available: "
                f"{sorted(k for k in aliases if k != 'all')}."
            )
        for package in aliases[key]:
            if package not in packages:
                packages.append(package)
    for package in packages:
        ok = nltk.download(package, quiet=quiet)
        if not ok:
            raise DictionaryResourceError(
                f"NLTK failed to download dictionary data package {package!r}."
            )


def _load_with_nltk_data(
    packages: tuple[str, ...], loader: Callable[[], T], *, download: bool
) -> T:
    try:
        return loader()
    except LookupError as initial_exc:
        if not download:
            raise _missing_resource_error(packages) from initial_exc

    globals()["download"](packages, quiet=True)
    try:
        return loader()
    except LookupError as retry_exc:  # pragma: no cover - provider failure
        raise _missing_resource_error(packages) from retry_exc


def _missing_resource_error(packages: tuple[str, ...]) -> DictionaryResourceError:
    commands = " ".join(packages)
    return DictionaryResourceError(
        "Required NLTK dictionary data are not installed: "
        f"{', '.join(packages)}. Run `python -m nltk.downloader {commands}` or "
        "call `teal.dictionaries.download_nltk(...)` explicitly."
    )


def _read_opinion_lexicon() -> tuple[list[str], list[str]]:
    from nltk.corpus import opinion_lexicon as corpus

    return list(corpus.positive()), list(corpus.negative())


def _read_vader_lexicon() -> dict[str, float]:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return {str(term): float(score) for term, score in analyzer.lexicon.items()}


def _read_sentiwordnet() -> dict[str, dict[str, float]]:
    from nltk.corpus import sentiwordnet as corpus

    positive: dict[str, float] = {}
    negative: dict[str, float] = {}
    objective: dict[str, float] = {}
    for senti_synset in corpus.all_senti_synsets():
        name = str(senti_synset.synset.name())
        positive[name] = float(senti_synset.pos_score())
        negative[name] = float(senti_synset.neg_score())
        objective[name] = float(senti_synset.obj_score())
    return {
        "positive": positive,
        "negative": negative,
        "objective": objective,
    }
