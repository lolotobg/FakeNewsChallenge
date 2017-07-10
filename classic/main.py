import sys
import numpy as np
import pickle
import os
from tqdm import tqdm

from itertools import chain
from sklearn.preprocessing.data import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, word_overlap_features
from feature_engineering import clean, get_tokenized_lemmas, gen_or_load_feats, has_feature
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import LABELS, score_submission, report_score

from utils.system import parse_params, check_version


ORIGINAL_LABELS = ['agree', 'disagree', 'discuss', 'unrelated']

def gen_or_load_texts(text_file, gen_fn):
    if os.path.isfile(text_file):
        with open(text_file, 'rb') as fp:
            feats = pickle.load(fp)
    else:
        feats = gen_fn()
        with open(text_file, 'wb') as fp:
            pickle.dump(feats, fp)
    return feats

def transform_labels_for_binary_relevance(labels):
    return list(map(lambda labelI: 0 if ORIGINAL_LABELS[labelI] != 'unrelated' else 1, labels))

def transform_features_for_binary_relevance(X_overlap, X_hand, X_refuting, firstStage):
    if (firstStage):
        return np.c_[X_hand, X_overlap]
    else:
        return np.c_[X_refuting, X_overlap]

def scaleMinMax(train_xs, test_xs):
    all_xs = list(train_xs) + list(test_xs)
    scaler = MinMaxScaler()
    scaler.fit(all_xs)
    train_xs_scaled = scaler.transform(train_xs)
    test_xs_scaled = scaler.transform(test_xs)
    return train_xs, test_xs


def generate_features(stances, all_articles, name):
    overlap_file = "features/overlap."+name+".npy"
    refuting_file = "features/refuting."+name+".npy"
    hand_file = "features/hand."+name+".npy"

    h, b, y = [],[],[]
    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(all_articles[stance['Body ID']])

    clean_headlines = []
    clean_bodies = []
    clean_lemmatized_headlines = []
    clean_lemmatized_bodies = []

    # Some feature is missing
    if not (has_feature(overlap_file) and has_feature(refuting_file) and has_feature(hand_file)):
        print("Cleaning texts")
        clean_headlines = gen_or_load_texts("features/clean_heads."+name+".p",
            lambda: list(map(clean, tqdm(h))))
        clean_bodies = gen_or_load_texts("features/clean_bodies."+name+".p",
            lambda: list(map(clean, tqdm(b))))
        print("Lemmatizing texts")
        clean_lemmatized_headlines = gen_or_load_texts("features/lemmas_heads."+name+".p",
            lambda: list(map(get_tokenized_lemmas, tqdm(clean_headlines))))
        clean_lemmatized_bodies = gen_or_load_texts("features/lemmas_bodies."+name+".p",
            lambda: list(map(get_tokenized_lemmas, tqdm(clean_bodies))))

    print("Computing 3 features")
    X_overlap = gen_or_load_feats(word_overlap_features,
        clean_lemmatized_headlines, clean_lemmatized_bodies, overlap_file)
    X_refuting = gen_or_load_feats(refuting_features,
        clean_lemmatized_headlines, clean_lemmatized_bodies, refuting_file)
    X_hand = gen_or_load_feats(hand_features, clean_headlines, clean_bodies, hand_file)

    return X_overlap,X_hand,X_refuting,y


def evaluate_double_stage_on_test():
    print("Evaluating 2-stage classifier on TEST dataset")

    # Load TRAIN data and all features on it
    train_dataset = DataSet()
    print("Computing features for TRAIN data")
    train_overlap, train_hand, train_refuting, train_all_ys = generate_features(
        train_dataset.stances, train_dataset.articles, 'train_dataset')
    train_stage1_ys = transform_labels_for_binary_relevance(train_all_ys)
    train_stage1_xs = transform_features_for_binary_relevance(train_overlap, train_hand, train_refuting, True)
    train_stage2_xs = transform_features_for_binary_relevance(train_overlap, train_hand, train_refuting, False)

    test_dataset = DataSet("competition_test")
    # Load TEST data and all features on it
    print("Computing features for TEST data")
    test_overlap, test_hand, test_refuting, test_all_ys = generate_features(
        test_dataset.stances, test_dataset.articles, 'test_dataset')
    test_stage1_ys = transform_labels_for_binary_relevance(test_all_ys)
    test_stage1_xs = transform_features_for_binary_relevance(test_overlap, test_hand, test_refuting, True)
    test_stage2_xs = transform_features_for_binary_relevance(test_overlap, test_hand, test_refuting, False)

    train_stage1_xs, test_stage1_xs = scaleMinMax(train_stage1_xs, test_stage1_xs)
    train_stage2_xs, test_stage2_xs = scaleMinMax(train_stage2_xs, test_stage2_xs)

    print("Training stage 1")
    classifier_stage1 = MLPClassifier(random_state=14128, hidden_layer_sizes=(60, 30))
    classifier_stage1.fit(train_stage1_xs, train_stage1_ys)

    train_xs_related = []
    train_ys_related = []
    for i, y in enumerate(train_stage1_ys):
        if y == 0: # related
            train_xs_related.append(train_stage2_xs[i])
            train_ys_related.append(train_all_ys[i])

    classifier_stage2 = MLPClassifier(random_state=14128, hidden_layer_sizes=(200))

    print("Training stage 2")
    classifier_stage2.fit(train_xs_related, train_ys_related)

    print("Classifying TEST data stage 1")
    labels = ['related', 'unrelated']
    relatedLabels = []
    predicted = [labels[int(a)] for a in classifier_stage1.predict(test_stage1_xs)]
    actual_first_stage = [labels[int(a)] for a in test_stage1_ys]
    print("First stage results")
    report_score(actual_first_stage, predicted, labels, relatedLabels)

    print("Classifying TEST data stage 2")
    labels = ORIGINAL_LABELS
    relatedLabels = labels[0:3]
    for i, label in enumerate(predicted):
        if label == 'related':
            new_pred = classifier_stage2.predict([test_stage2_xs[i]])[0]
            predicted[i] = labels[int(new_pred)]
    actual = [labels[int(a)] for a in test_all_ys]
    print("Both stages results")
    report_score(actual, predicted, labels, relatedLabels)


def evaluate_single_stage_on_test():
    print("Evaluating 1-stage classifier on TEST dataset")

    # Load TRAIN data and all features on it
    train_dataset = DataSet()
    print("Computing features for TRAIN data")
    train_overlap, train_hand, train_refuting, train_ys = generate_features(
        train_dataset.stances, train_dataset.articles, 'train_dataset')
    train_xs = np.c_[train_overlap, train_hand, train_refuting]

    test_dataset = DataSet("competition_test")
    # Load TEST data and all features on it
    print("Computing features for TEST data")
    test_overlap, test_hand, test_refuting, test_ys = generate_features(
        test_dataset.stances, test_dataset.articles, 'test_dataset')
    test_xs = np.c_[test_overlap, test_hand, test_refuting]

    train_xs, test_xs = scaleMinMax(train_xs, test_xs)

    classifier = MLPClassifier(random_state=14128)
    print("Training")
    classifier.fit(train_xs, train_ys)

    print("Classifying")
    labels = ORIGINAL_LABELS
    predicted = [labels[int(a)] for a in classifier.predict(test_xs)]
    actual = [labels[int(a)] for a in test_ys]

    relatedLabels = labels[0:3]
    report_score(actual, predicted, labels, relatedLabels)


def evaluate_single_stage_on_train():
    labels = ORIGINAL_LABELS
    relatedLabels = labels[0:3]
    print("Evaluating 1-stage classifier on TRAIN dataset with 10-fold cross-validation")

    train_dataset = DataSet()
    folds = kfold_split(train_dataset, n_folds=10)
    fold_stances = get_stances_for_folds(train_dataset, folds)

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    for fold in fold_stances:
        print("Computing features for fold "+ str(fold))
        fold_overlap, fold_hand, fold_refuting, ys[fold] = generate_features(
            fold_stances[fold], train_dataset.articles, str(fold))
        Xs[fold] = np.c_[fold_overlap, fold_hand, fold_refuting]

    all_data = [item for fold in list(Xs.values()) for item in fold]
    scaler = MinMaxScaler()
    scaler.fit(all_data)
    for fold in fold_stances:
        Xs[fold] = scaler.transform(Xs[fold])

    print("Classifying folds")

    average_score = 0
    total_count = 0

    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = RandomForestClassifier(n_estimators=200, random_state=14128)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted, labels, relatedLabels)
        max_fold_score, _ = score_submission(actual, actual, labels, relatedLabels)

        score = fold_score / max_fold_score
        average_score += score

        print("Score for fold "+ str(fold) + " = " + str(score))

    average_score /= len(fold_stances)
    print("Average score " + str(average_score))


if __name__ == "__main__":
    check_version()
    parse_params()

    #evaluate_double_stage_on_test()
    #evaluate_single_stage_on_test()
    evaluate_single_stage_on_train()
