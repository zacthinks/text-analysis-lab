"""External dictionary-provider adapters."""

from text_analysis_lab.dictionaries.providers.afinn import afinn
from text_analysis_lab.dictionaries.providers.nltk import (
    DictionaryResourceError,
    download as download_nltk,
    opinion_lexicon as nltk_opinion_lexicon,
    sentiwordnet as nltk_sentiwordnet,
    vader as nltk_vader,
)

__all__ = [
    "DictionaryResourceError",
    "afinn",
    "download_nltk",
    "nltk_opinion_lexicon",
    "nltk_sentiwordnet",
    "nltk_vader",
]
