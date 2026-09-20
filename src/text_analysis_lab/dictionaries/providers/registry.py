"""Reload external dictionary resources from their provider references."""

from __future__ import annotations

from text_analysis_lab.dictionaries.polarity import PolarityDictionary
from text_analysis_lab.dictionaries.providers.nltk import DictionaryResourceError
from text_analysis_lab.dictionaries.source import DictionarySource
from text_analysis_lab.dictionaries.valence import ValenceDictionary


def load_dictionary_source(
    source: DictionarySource,
) -> PolarityDictionary | ValenceDictionary:
    """Load one supported external dictionary without implicit downloads."""

    provider = source.provider.lower()
    resource = source.resource.lower()
    if provider == "nltk":
        from text_analysis_lab.dictionaries.providers import nltk as nltk_provider

        if resource == "opinion_lexicon":
            return nltk_provider.opinion_lexicon(download=False)
        if resource == "vader":
            return nltk_provider.vader(download=False)
        if resource == "sentiwordnet":
            return nltk_provider.sentiwordnet(download=False)
    elif provider == "afinn" and resource == "afinn":
        from text_analysis_lab.dictionaries.providers.afinn import afinn

        return afinn(
            language=str(source.parameters.get("language", "en")),
            emoticons=bool(source.parameters.get("emoticons", False)),
        )

    raise DictionaryResourceError(
        f"TeAL does not know how to reload external dictionary source "
        f"{source.provider!r}/{source.resource!r}. Save the dictionary in the "
        "operator assets or recreate it as a user-supplied dictionary."
    )
