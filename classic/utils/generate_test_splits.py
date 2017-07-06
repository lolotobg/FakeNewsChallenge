import random
import os
from collections import defaultdict

def get_shuffled_training_set(dataset):
    r = random.Random()
    r.seed(1489215)
    article_ids = list(dataset.articles.keys())  # get a list of article ids
    r.shuffle(article_ids)  # and shuffle that list
    return article_ids

def kfold_split(dataset, n_folds = 10):
    training_ids = get_shuffled_training_set(dataset)
    item_count = len(training_ids)
    folds = []
    for k in range(n_folds):
        beginI = int(k * item_count / n_folds)
        endI = int((k + 1) * item_count / n_folds)
        folds.append(training_ids[beginI:endI])
    return folds

def get_stances_for_folds(dataset, folds):
    stances_folds = defaultdict(list)
    for stance in dataset.stances:
        fold_id = 0
        for fold in folds:
            if stance['Body ID'] in fold:
                stances_folds[fold_id].append(stance)
            fold_id += 1

    return stances_folds
