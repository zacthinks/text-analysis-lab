"""Content-analysis dictionary specifications and external-resource adapters."""

from text_analysis_lab.dictionaries.catalog import catalog
from text_analysis_lab.dictionaries.dictionary import Dictionary, ValueType
from text_analysis_lab.dictionaries.polarity import PolarityDictionary
from text_analysis_lab.dictionaries.provenance import DictionaryProvenance
from text_analysis_lab.dictionaries.source import DictionarySource
from text_analysis_lab.dictionaries.providers import (
    DictionaryResourceError,
    afinn,
    download_nltk,
    nltk_opinion_lexicon,
    nltk_sentiwordnet,
    nltk_vader,
)
from text_analysis_lab.dictionaries.valence import ValenceDictionary

# Friendly aliases for the standard resources. Provider-prefixed names remain
# available when provenance needs to be explicit in teaching/examples.
hu_liu = nltk_opinion_lexicon
vader = nltk_vader
sentiwordnet = nltk_sentiwordnet

__all__ = [
    "Dictionary",
    "DictionaryProvenance",
    "DictionaryResourceError",
    "DictionarySource",
    "PolarityDictionary",
    "ValenceDictionary",
    "ValueType",
    "afinn",
    "catalog",
    "download_nltk",
    "hu_liu",
    "nltk_opinion_lexicon",
    "nltk_sentiwordnet",
    "nltk_vader",
    "sentiwordnet",
    "vader",
]
