from reader import *
from writer import *
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.append("../")

def extractFeatures(train, test, task, analyzer='word', max_features=1000, ngram_range=(2, 4), stop_words='english'):
    if task == 'class':
        test_tweets_list = []
        train_tweets_list = []

        test_emotions_list = []
        train_emotions_list = []

        for file in test:
            test_tweets, test_emotion, test_labels, test_ids = readTweetsOfficial(file, task=task)
            test_tweets_list.extend(test_tweets)
            test_emotions_list.extend(test_emotion)

        for file in train:
            train_tweets, train_emotion, train_labels, train_ids = readTweetsOfficial(file, task=task)
            train_tweets_list.extend(train_tweets)
            train_emotions_list.extend(train_emotion)

        train_features, test_features, vocab = featTransform(train_tweets_list, test_tweets_list, analyzer=analyzer, max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
        
        return train_features, train_emotions_list, test_features, test_emotions_list
    
    elif task == 'reg':
        test_tweets_list = []
        train_tweets_list = []

        test_labels_list = []
        train_labels_list = []

        test_features_list = []
        train_features_list = []

        for file in test:
            test_tweets, test_emotion, test_labels, test_ids = readTweetsOfficial(file)
            test_tweets_list.append(test_tweets)
            test_labels_list.append(test_labels)

        for file in train:
            train_tweets, train_emotion, train_labels, train_ids = readTweetsOfficial(file, task=task)
            train_tweets_list.append(train_tweets)
            train_labels_list.append(train_labels)

        for i in range(4):
            train_features, test_features, vocab = featTransform(train_tweets_list[i], test_tweets_list[i], analyzer=analyzer, max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
            test_features_list.append(test_features)
            train_features_list.append(train_features)

        return train_features_list, train_labels_list, test_features_list, test_labels_list


def featTransform(train_tweets, test_tweets, analyzer, max_features, ngram_range, stop_words):
    # max_features=100, ngram_range=(1, 4), stop_words='english'
    TfidfV = TfidfVectorizer(analyzer=analyzer, max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
    TfidfV.fit(train_tweets)
    # print(TfidfV.vocabulary_)            
    train_features = TfidfV.transform(train_tweets)
    test_features = TfidfV.transform(test_tweets)
    train_features = train_features.todense()
    test_features = test_features.todense()
    feat_train = []
    feat_test = []
    for i, tweet in enumerate(train_tweets):
        if '!' in tweet:
            feat_train.append([1])
        else:
            feat_train.append([0])

    for j, tweets in enumerate(test_tweets):
        if '!' in tweets:
            feat_test.append([1])
        else:
            feat_test.append([0])
    train_features = np.append(train_features, feat_train, 1)
    test_features = np.append(test_features, feat_test, 1)

    # print(train_features)
    # print(test_features)
    return train_features, test_features, TfidfV.vocabulary_
