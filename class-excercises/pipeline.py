import copy
from csv import DictReader

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.svm import SVC

from features import SentenceLength, BagOfTfIDF, WordOverlap
from features import POS, NER
from features import ToMatrix
from preprocess import TokenizedLemmas

from scorer import *


def get_dataset():
    # use FNC's function for reading dataset.
    training_set_dicts = load_dataset("../datasets/train_stances.csv")

    # read the bodies
    bodies = DictReader(open("../datasets/train_bodies.csv", encoding="utf8"))
    bodies_dict = {i["Body ID"]: i["articleBody"] for i in bodies}
    for instance in training_set_dicts:
        instance['articleBody'] = bodies_dict[instance["Body ID"]]

        instance['features'] = {}

    # print(training_set_dicts[0])

    return training_set_dicts


def run_classifier(test, train):

    pipeline = Pipeline([('preprocess_lemmas', TokenizedLemmas()),
                                  # ('sent_len', SentenceLength()),
                                  # ('tfidf', BagOfTfIDF(train)),
                                  # ('pos', POS()),
                                  # ('ner', NER()),
                                  ('word_overlap', WordOverlap()),
                                  ('transform', ToMatrix()),
                                  ('norm', MinMaxScaler()),
                                  ('clf', SVC())])

    print("Started pipeline ...")

    true_labels = [instance['Stance'] for instance in train]
    pipeline.fit(train, true_labels)

    print("Finished training.")

    # return the predicted labels of the test set
    return pipeline.predict(test)


def test_cross_validation():
    X = get_dataset()[:100]

    # 1. Run cross-validation, using KFold class from sklearn
    k_folds = KFold(n_splits=10, random_state=42)
    scores = 0  # this will store the sum of scores of each split
    relative_scores = 0

    # 2. Iterate over the 5 splits of the data
    for train_index, test_index in k_folds.split(X):
        X_train = [X[i] for i in train_index]
        X_test = [X[i] for i in test_index]

        #   2.1. Run a classifier with the fold
        predicted_labels = run_classifier(train=X_train, test=X_test)

        #   2.2. Get the score of the prediction, using get_fnc_score
        score, rate_null = get_fnc_score(X_test, predicted_labels)
        print("Score {0:.5f}% of max and {1:.5f} relative to const unreleated".format(score, rate_null))
        # print_confusion_matrix(cm)
        scores += score
        relative_scores += rate_null

    # 2.3. Get the average score, achieved from the cross-validation
    print("Average score for {0} folds is {1:.5f}% and relative imporvement {2:.5f}".format(
        k_folds.n_splits, scores / k_folds.n_splits, relative_scores / k_folds.n_splits))


def get_fnc_score(X_test, predicted_labels):
    """Helper function to get FNC score"""
    X_test_pred = copy.deepcopy(X_test)
    for x, pred in zip(X_test_pred, predicted_labels):
        x['Stance'] = pred

    # return the submission score of FNC system
    score, confusion_matrix = score_submission(X_test, X_test_pred)
    null_score, max_score = score_defaults(X_test)
    return [float(score) / (max_score + 0.0000001), float(score) / (null_score + 0.0000001)]

if __name__ == "__main__":
    test_cross_validation()

