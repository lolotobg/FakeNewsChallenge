import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm


_wnl = nltk.WordNetLemmatizer()

_refuting_words = [
    'fake',
    'fraud',
    'hoax',
    'false',
    'deny', 'denies',
    # 'refute',
    'not',
    'despite',
    'nope',
    'doubt', 'doubts',
    'bogus',
    'debunk',
    'pranks',
    'retract'
]
_refuting_words_set = set(_refuting_words)


def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def has_feature(feature_file):
    return os.path.isfile(feature_file)

def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)
    return np.load(feature_file)


def word_overlap_features(clean_lemmatized_headlines, clean_lemmatized_bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(clean_lemmatized_headlines, clean_lemmatized_bodies))):
        headline_set = set(headline)
        features = [
            len(headline_set.intersection(body)) / float(len(headline_set.union(body)))]
        X.append(features)
    return X


def refuting_features(clean_lemmatized_headlines, _):

    X = []
    for headline_lemmas in tqdm(clean_lemmatized_headlines):
        headline_set = set(headline_lemmas)
        features = [1 if word in headline_set else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(clean_lemmatized_headlines, clean_lemmatized_bodies):
    def calculate_polarity(tokens):
        return sum([t in _refuting_words_set for t in tokens]) % 2

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(clean_lemmatized_headlines, clean_lemmatized_bodies))):
        features = []
        features.append(calculate_polarity(headline))
        features.append(calculate_polarity(body))
        X.append(features)
    return np.array(X)


def ngrams(tokens, n):
    return chargrams(tokens, n)


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, headline_no_stops, clean_body, clean_body_100, clean_body_255, size):
    grams = [' '.join(x) for x in chargrams(headline_no_stops, size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in clean_body_100:
            grams_hits += 1
            grams_early_hits += 1
            grams_first_hits += 1
        elif gram in clean_body_255:
            grams_hits += 1
            grams_early_hits += 1
        elif gram in clean_body:
            grams_hits += 1

    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, headline_tokens, clean_body, _clean_body_100, clean_body_255, size):
    grams = [' '.join(x) for x in ngrams(headline_tokens, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in clean_body_255:
            grams_hits += 1
            grams_early_hits += 1
        elif gram in clean_body:
            grams_hits += 1

    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(clean_headlines, clean_bodies):

    def binary_co_occurence(headline_tokens, clean_body, _clean_body_100, clean_body_255):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in headline_tokens:
            if headline_token in clean_body_255:
                bin_count += 1
                bin_count_early += 1
            elif headline_token in clean_body:
                bin_count += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline_tokens, clean_body, _clean_body_100, _clean_body_255):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(headline_tokens):
            if headline_token in clean_body:
                bin_count += 1
                bin_count_early += 1 # Haha, bin_count_early == bin_count
        return [bin_count, bin_count_early]

    def count_grams(headline_tokens, clean_body, clean_body_100, clean_body_255):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        headline_no_stops = " ".join(remove_stopwords(headline_tokens))
        features = []
        features = append_chargrams(features, headline_no_stops, clean_body, clean_body_100, clean_body_255, 2)
        features = append_chargrams(features, headline_no_stops, clean_body, clean_body_100, clean_body_255, 8)
        features = append_chargrams(features, headline_no_stops, clean_body, clean_body_100, clean_body_255, 4)
        features = append_chargrams(features, headline_no_stops, clean_body, clean_body_100, clean_body_255, 16)
        features = append_ngrams(features, headline_tokens, clean_body, clean_body_100, clean_body_255, 2)
        features = append_ngrams(features, headline_tokens, clean_body, clean_body_100, clean_body_255, 3)
        features = append_ngrams(features, headline_tokens, clean_body, clean_body_100, clean_body_255, 4)
        features = append_ngrams(features, headline_tokens, clean_body, clean_body_100, clean_body_255, 5)
        features = append_ngrams(features, headline_tokens, clean_body, clean_body_100, clean_body_255, 6)
        return features

    X = []
    for i, (clean_headline, clean_body) in tqdm(enumerate(zip(clean_headlines, clean_bodies))):
        headline_tokens = clean_headline.split(' ')
        clean_body_100 = clean_body[:100]
        clean_body_255 = clean_body[:255]
        X.append(binary_co_occurence(headline_tokens, clean_body, clean_body_100, clean_body_255)
                 + binary_co_occurence_stops(headline_tokens, clean_body, clean_body_100, clean_body_255)
                 + count_grams(headline_tokens, clean_body, clean_body_100, clean_body_255))

    return X
