from nltk.data import load
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk, tag
from sklearn.base import TransformerMixin


class Feature(TransformerMixin):
    """Feature Interface."""
    def fit(self, X, y=None, **fit_params):
        return self


class ToMatrix(Feature):
    """Transforms the features dict to a matrix"""
    def transform(self, X):
        final_X = []

        feature_names = X[0]['features'].keys()

        for instance in X:
            sent_vector = []
            for feat in feature_names:
                if isinstance(instance['features'][feat], list):
                    sent_vector += instance['features'][feat]
                else:
                    sent_vector.append(instance['features'][feat])
            final_X.append(sent_vector)

        return final_X


class BagOfTfIDF(Feature):
    """Adds Bag of TF-IDF scores of 1/2/3-grams."""

    def __init__(self, training):
        self.vectorizer = TfidfVectorizer(analyzer="word",
                                          tokenizer=None,
                                          preprocessor=None,
                                          ngram_range=(1, 1),
                                          min_df=10,
                                          max_df=0.4,
                                          stop_words="english",
                                          max_features=100)
        training_text = [instance['articleBody'] for instance in training]
        self.vectorizer.fit_transform(training_text)

    def transform(self, X):
        for instance in X:
            instance['features']['bag_tfidf'] = \
                self.vectorizer.transform([instance['articleBody']]).toarray().tolist()[0]
        return X


class SentenceLength(Feature):
    """Add some statistics about text's length."""

    def transform(self, X):
        """
        Add number of tokens and number of chars in the bodies as separate features to each instances' dictionary
        of features.
        """
        for instance in X:
            instance['features']['tokens_num'] = len(
                word_tokenize(instance['articleBody']))  # this counts the punctuation, too
            instance['features']['text_len'] = len(instance['articleBody'])
        return X


class WordOverlap(Feature):

    def transform(self, X):
        """
        Use the lemmatized tokens of the headline and the body and add as a feature the number of the overlapping
        words (intersection) in the headline and the body.
        Normalize this by the total number of the tokens in the body and the headline (union).
        """
        for instance in X:
            instance['features']['overlapping_words'] = len(set(instance['headline_lemmas'])
                                                            .intersection(instance['body_lemmas'])) / \
                                                        float(len(set(set(instance['headline_lemmas'])).
                                                                  union(instance['body_lemmas'])))
        return X


class POS(Feature):
    """Adds a vector of POS tag counts."""

    def __init__(self, normalized=False):
        self.normalized = normalized

        # get the names of the tags and put them in a dictionary (key is index in the resulting vector)
        tagdict = load('help/tagsets/upenn_tagset.pickle')
        tags_keys = tagdict.keys()
        self.tags = {}
        for i, tag in enumerate(tags_keys):
            self.tags[tag] = i

    def transform(self, X):
        """
        Add as a feature the number POS tag occurrences in the POS tags of the body.
        This will end up in a feature vector.
        """
        for instance in X:
            # POS-tagging requires tokenized text.
            # tokenizing is a slow operation, do you see an optimization here?
            tokenized = word_tokenize(instance['articleBody'])
            pos_tags = pos_tag(tokenized)

            # fill a zero-vector with number of each tag's occurrence
            tag_vector = [0 for _ in range(len(self.tags))]
            for word, tag in pos_tags:
                if tag not in self.tags:
                    continue
                tag_vector[self.tags[tag]] += 1

            # can normalize by number of words in the text
            if self.normalized:
                instance['features']['pos'] = [tag / len(tokenized) for tag in tag_vector]
            else:
                instance['features']['pos'] = tag_vector
        return X


class NER(Feature):
    """Adds NEs count"""

    def transform(self, X):
        """
        Add as a feature the number of Named Entities encountered in the body.
        """
        for instance in X:
            # NER requires pos tags as an input:
            # POS-tagging is a slow operation, do you see an optimization here?
            tokens = word_tokenize(instance['articleBody'])
            parse_tree = ne_chunk(tag.pos_tag(tokens), binary=True)  # POS tagging before chunking!

            named_entities = []
            # get the NEs in the parsed tree
            for subtree in parse_tree.subtrees():
                if subtree.label() == 'NE':
                    named_entities.append(subtree)

            # get the number of the NEs
            instance['features']['ner_count_nltk'] = len(named_entities)
        return X