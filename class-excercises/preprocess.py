from nltk import WordNetLemmatizer, word_tokenize
import re
from sklearn import feature_extraction
from features import Feature

_wnl = WordNetLemmatizer()


class TokenizedLemmas(Feature):
    def transform(self, X):
        """
        Pre-process the headline and the body of the instances.
        Get the list of lemmatized tokens in them.
        Add them to the instance's dictionary, keeping the original text.
        """
        for instance in X:
            instance['body_lemmas'] = _get_tokenized_lemmas(instance['articleBody'])
            instance['headline_lemmas'] = _get_tokenized_lemmas(instance['Headline'])
        return X


def _remove_stopwords(tokens_list):
    """ Removes stopwords from a list of tokens/"""
    return [w for w in tokens_list if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def _normalize_word(word):
    """ Lemmatize the given word."""
    return _wnl.lemmatize(word).lower()


def _get_tokenized_lemmas(text):
    """ Return a list of lemmatized tokens, found in the given text."""
    return [_normalize_word(t) for t in word_tokenize(text)]


def _clean(text):
    """ Cleans the text: Lowercasing, trimming, removing non-alphanumeric"""
    return " ".join(re.findall(r'\w+', text, flags=re.UNICODE)).lower()
