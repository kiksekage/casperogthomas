import sys
sys.path.append("../")

from reader import *
from writer import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer


def extractFeatures(train, test):
    test_tweets_list = []
    train_tweets_list = []

    test_labels_list = []
    train_labels_list = []

    test_features_list = []
    train_features_list = []

    for file in test:
        test_tweets, test_emotion, test_labels, test_ids = readTweetsOfficial(
            file)
        test_tweets_list.append(test_tweets)
        test_labels_list.append(test_labels)

    for file in train:
        train_tweets, train_emotion, train_labels, train_ids = readTweetsOfficial(
            file)
        train_tweets_list.append(train_tweets)
        train_labels_list.append(train_labels)

    for i in range(4):
        train_features, test_features, vocab = featTransform(
            train_tweets_list[i], test_tweets_list[i])
        test_features_list.append(test_features)
        train_features_list.append(train_features)

    return train_features_list, train_labels_list, test_features_list, test_labels_list


def featTransform(train_tweets, test_tweets):
    # max_features=100, ngram_range=(1, 4), stop_words='english'
    TfidfV = TfidfVectorizer(max_features=1000, stop_words='english')
    TfidfV.fit(train_tweets)
    #print(TfidfV.vocabulary_)
    train_features = TfidfV.transform(train_tweets)
    test_features = TfidfV.transform(test_tweets)
    #print(train_features)
    #print(test_features)
    return train_features, test_features, TfidfV.vocabulary


def model_train(train_features, train_labels):
    model = RandomForestRegressor()
    model.fit(train_features, train_labels)
    #print(model.get_params())
    return model

def predict(model, test_features):
    preds = model.predict(test_features)
    #print(preds)
    return preds

if __name__ == '__main__':
    pred_dict = {0 : 'anger_pred.txt', 1 : 'fear_pred.txt', 2 : 'joy_pred.txt', 3 : 'sadness_pred.txt'}
    pred_fold = 'preds/'

    if sys.argv[1] == '17':
        fp = "../testkode/data17/"

        train = [fp + "train/anger-ratings-0to1.train.txt", fp + "train/fear-ratings-0to1.train.txt",
                 fp + "train/joy-ratings-0to1.train.txt", fp + "train/sadness-ratings-0to1.train.txt"]
        test = [fp + "test/anger-ratings-0to1.test.target.txt", fp + "test/fear-ratings-0to1.test.target.txt",
                fp + "test/joy-ratings-0to1.test.target.txt", fp + "test/sadness-ratings-0to1.test.target.txt"]
        dev = [fp + "dev/anger-ratings-0to1.dev.txt", fp + "dev/fear-ratings-0to1.dev.txt",
               fp + "dev/joy-ratings-0to1.dev.txt", fp + "dev/sadness-ratings-0to1.dev.txt"]

        train_features, train_labels, test_features, test_labels = extractFeatures(
            train, test)  # aendr test til dev

        for i, emotion in enumerate(train_features):
            model = model_train(emotion, train_labels[i])
            predictions = predict(model, test_features[i])
       
            #print(predictions)
            #print(train_labels)

            printPredsToFileReg(test[i], pred_fold + pred_dict[i], predictions)  # aendr test til dev

    elif sys.argv[1] == "18":
        fp = "../testkode/data18/2018-EI-reg-En-train/"
        fp_test = "../testkode/data17/"

        train = [fp + "2018-EI-reg-En-anger-train.txt", fp + "2018-EI-reg-En-fear-train.txt",
                 fp + "2018-EI-reg-En-joy-train.txt", fp + "2018-EI-reg-En-sadness-train.txt"]
        test = [fp_test + "test/anger-ratings-0to1.test.target.txt", fp_test + "test/fear-ratings-0to1.test.target.txt",
                fp_test + "test/joy-ratings-0to1.test.target.txt", fp_test + "test/sadness-ratings-0to1.test.target.txt"]
        dev = [fp + "dev/anger-ratings-0to1.dev.txt", fp + "dev/fear-ratings-0to1.dev.txt",
               fp + "dev/joy-ratings-0to1.dev.txt", fp + "dev/sadness-ratings-0to1.dev.txt"]

        train_features, train_labels, test_features, test_labels = extractFeatures(
            train, test)  # aendr test til dev

        for i, emotion in enumerate(train_features):
            model = model_train(emotion, train_labels[i])
            predictions = predict(model, test_features[i])
       
            #print(predictions)
            #print(train_labels)

            printPredsToFileReg(test[i], pred_fold + pred_dict[i], predictions)  # aendr test til dev
