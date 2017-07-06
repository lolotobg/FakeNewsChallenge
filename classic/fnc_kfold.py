import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, word_overlap_features
from feature_engineering import clean, get_tokenized_lemmas, gen_or_load_feats, has_feature
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import LABELS, score_submission, report_score

from utils.system import parse_params, check_version


def generate_features(stances,dataset,name):
    overlap_file = "features/overlap."+name+".npy"
    refuting_file = "features/refuting."+name+".npy"
    polarity_file = "features/polarity."+name+".npy"
    hand_file = "features/hand."+name+".npy"

    h, b, y = [],[],[]
    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    clean_headlines = []
    clean_bodies = []
    clean_lemmatized_headlines = []
    clean_lemmatized_bodies = []

    # Some feature missing
    if not (has_feature(overlap_file) and has_feature(refuting_file) and has_feature(polarity_file) and has_feature(hand_file)):
        print("Cleaning and lemmatizing text")
        clean_headlines = list(map(clean, h))
        clean_bodies = list(map(clean, b))
        clean_lemmatized_headlines = list(map(get_tokenized_lemmas, clean_headlines))
        clean_lemmatized_bodies = list(map(get_tokenized_lemmas, clean_bodies))

    X_overlap = gen_or_load_feats(word_overlap_features,
        clean_lemmatized_headlines, clean_lemmatized_bodies, overlap_file)
    X_refuting = gen_or_load_feats(refuting_features,
        clean_lemmatized_headlines, clean_lemmatized_bodies, refuting_file)
    X_polarity = gen_or_load_feats(polarity_features,
        clean_lemmatized_headlines, clean_lemmatized_bodies, polarity_file)
    X_hand = gen_or_load_feats(hand_features, clean_headlines, clean_bodies, hand_file)

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

def evaluate_on_test():
    train_dataset = DataSet()

    # Load/Precompute features on all of the TRAIN data
    print("Computing features for TRAIN data")
    train_xs, train_ys = generate_features(train_dataset.stances, train_dataset, 'train_dataset')

    # Load competition test dataset
    test_dataset = DataSet("competition_test")

    # Load/Precompute features on all of the TEST data
    print("Computing features for competition test data")
    test_xs, test_ys = generate_features(test_dataset.stances, test_dataset, 'test_dataset')

    print("Training on the whole train dataset with 200 estimators")
    clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
    clf.fit(train_xs, train_ys)

    print("Classifying competition test data")
    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in clf.predict(test_xs)]
    actual = [LABELS[int(a)] for a in test_ys]

    report_score(actual,predicted)


if __name__ == "__main__":
    check_version()
    parse_params()

    train_dataset = DataSet()
    folds = kfold_split(train_dataset, n_folds=10)
    fold_stances = get_stances_for_folds(train_dataset, folds)

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    for fold in fold_stances:
        print("Computing features for fold "+ str(fold))
        Xs[fold],ys[fold] = generate_features(fold_stances[fold], train_dataset, str(fold))

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

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score / max_fold_score
        average_score += score

        print("Score for fold "+ str(fold) + " - " + str(score))

    average_score /= len(fold_stances)
    print("Average score " + str(average_score))

    # evaluate_on_test()
