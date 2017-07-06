# nlp-su
This repository contains exercises for the NLP summer course in Sofia University.

This branch's task is to implement a classifier, using the data, provided by FNC.
It explores preprocessing, language features, using the nltk library, cross-validation and scoring - a full NLP classification pipeline.

Structure of the classification task:

pipeline.py:

test_cross_validation -> the main function, running the task
1. Reading data : get_dataset - Reading the data with the help of FNC's implementation, adding the text of bodies ourselves
2. Splitting the data for cross-validation evaluation
3. Training and testing a classifier with each fold: run_classifier
3.1. Define a pipeline of features, which will be extracted from the data, this will produce a matrix NXM with features for the instances:

Pipeline([('preprocess_lemmas', TokenizedLemmas()),
          ('sent_len', SentenceLength()),
          ('tfidf', BagOfTfIDF(train)),
          ('pos', POS()),
          ('ner', NER()),
          ('word_overlap', WordOverlap()),
          ('transform', ToMatrix()),
          ('norm', MinMaxScaler())])

TokenizedLemmas: preprocess text of the instances to produce lemmatized tokens
SentenceLength, BagOfTfIDF, NER, POS, WordOverlap - various NLP features
ToMatrix: transforms the dictionary of features into a matrix
MinMaxScaler: normalizes the arrays to the 0-1 scale


3.2. Run features on the data, collecting features in the "features" dictionary of each instance
3.3. Train classifier on collected features
3.4. Test classifier on collected features
4. Evaluate the predicted labels for the fold with the get_fnc_score helper function
