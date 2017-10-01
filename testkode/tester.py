import sys
sys.path.append("../")

from reader import *
from writer import *
from eval import *
#from isabelle_eksempler.BridgesML.ex2_scikit import *
from sklearn.neighbors import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer

def extractFeatures(train, test):
    test_tweets_list = []
    train_tweets_list = []

    test_labels_list =[]
    train_labels_list =[]

    for file in test:
        test_tweets, test_emotion, test_labels, test_ids = readTweetsOfficial(file)
        test_tweets_list.extend(test_tweets)
        test_labels_list.extend(test_emotion)
    
    for file in train:
        train_tweets, train_emotion, train_labels, train_ids = readTweetsOfficial(file)
        train_tweets_list.extend(train_tweets)
        train_labels_list.extend(train_emotion)

    train_features, test_features, vocab = featTransform(train_tweets_list, test_tweets_list)

    return train_features, train_labels_list, test_features, test_labels_list

def featTransform(train_tweets, test_tweets):
    TfidfV = TfidfVectorizer(max_features=1000, stop_words='english') #max_features=100, ngram_range=(1, 4), stop_words='english'
    TfidfV.fit(train_tweets)
    #print(TfidfV.vocabulary_)
    train_features = TfidfV.transform(train_tweets)
    test_features = TfidfV.transform(test_tweets)
    #print(train_features)
    #print(test_features)
    return train_features, test_features, TfidfV.vocabulary

def model_train(feats_train, labels):
    # s(f(x), g(x)) + loss function handled by this model
    model = NearestCentroid()
    model.fit(feats_train, labels)
    return model

def predict(model, features_test):
    """Find the most compatible output class given the input `x` and parameter `theta`"""
    preds = model.predict(features_test)
    #preds_prob = model.predict_proba(features_test)  # probabilities instead of classes
    #print(preds)
    return preds

if __name__ == '__main__':
    if sys.argv[1] == '17':
        fp = "../testkode/data17/"
        train = [fp + "train/anger-ratings-0to1.train.txt", fp + "train/fear-ratings-0to1.train.txt", fp + "train/joy-ratings-0to1.train.txt", fp + "train/sadness-ratings-0to1.train.txt"]
        test = [fp + "test/anger-ratings-0to1.test.target.txt", fp + "test/fear-ratings-0to1.test.target.txt", fp + "test/joy-ratings-0to1.test.target.txt", fp + "test/sadness-ratings-0to1.test.target.txt"]
        dev = [fp + "dev/anger-ratings-0to1.dev.txt", fp + "dev/fear-ratings-0to1.dev.txt", fp + "dev/joy-ratings-0to1.dev.txt", fp + "dev/sadness-ratings-0to1.dev.txt"]
        pred = "../testkode/preds.txt"

        train_features, train_labels, test_features, test_labels = extractFeatures(train, test) #ændr test til dev

        model = model_train(train_features, train_labels)
        predictions = predict(model, test_features)
        
        #print(predictions)
        #print(train_labels)

        printPredsToFile(test, pred, predictions) #ændr test til dev
        eval(pred)
    else:
        fp = "../testkode/data18/2018-EI-reg-En-train/"
        fp_test = "../testkode/data17/"
        train = [fp + "2018-EI-reg-En-anger-train.txt", fp + "2018-EI-reg-En-fear-train.txt", fp + "2018-EI-reg-En-joy-train.txt", fp + "2018-EI-reg-En-sadness-train.txt"]
        test = [fp_test + "test/anger-ratings-0to1.test.target.txt", fp_test + "test/fear-ratings-0to1.test.target.txt", fp_test + "test/joy-ratings-0to1.test.target.txt", fp_test + "test/sadness-ratings-0to1.test.target.txt"]
        pred = "../testkode/preds.txt"

        train_features, train_labels, test_features, test_labels = extractFeatures(train, test) #ændr test til dev

        model = model_train(train_features, train_labels)
        predictions = predict(model, test_features)
        
        #print(predictions)
        #print(train_labels)

        printPredsToFile(test, pred, predictions) #ændr test til dev
        eval(pred)
