"""External dictionary-provider adapters."""

from text_analysis_lab.dictionaries.providers.afinn import afinn
from text_analysis_lab.dictionaries.providers.nltk import (
    DictionaryResourceError,
)
from text_analysis_lab.dictionaries.providers.nltk import (
    download as download_nltk,
)
from text_analysis_lab.dictionaries.providers.nltk import (
    opinion_lexicon as nltk_opinion_lexicon,
)
from text_analysis_lab.dictionaries.providers.nltk import (
    sentiwordnet as nltk_sentiwordnet,
)
from text_analysis_lab.dictionaries.providers.nltk import (
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
