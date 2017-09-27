import sys
sys.path.append("../")

from reader import *
from writer import *
#from isabelle_eksempler.BridgesML.ex2_scikit import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

def extractFeatures(train, test):
    test_emotion, test_tweets, test_labels, test_ids = readTweetsOfficial(test)
    train_emotion, train_tweets, train_labels, train_ids = readTweetsOfficial(train)

    train_features, test_features, vocab = featTransform(train_tweets, test_tweets)

    return train_features, train_labels, test_features, test_labels

def featTransform(train_tweets, test_tweets):
    cv = CountVectorizer(max_features=100, stop_words='english') #max_features=100, ngram_range=(1, 4), stop_words='english'
    cv.fit(train_tweets)
    #print(cv.vocabulary_) #lige kommenteret printet ud
    train_features = cv.transform(train_tweets)
    test_features = cv.transform(test_tweets)
    #print(features_train)
    return train_features, test_features, cv.vocabulary

def model_train(feats_train, labels):
    # s(f(x), g(x)) + loss function handled by this model
    model = LogisticRegression(penalty='l2')
    model.fit(feats_train, labels)
    return model

def predict(model, features_test):
    """Find the most compatible output class given the input `x` and parameter `theta`"""
    preds = model.predict(features_test)
    #preds_prob = model.predict_proba(features_test)  # probabilities instead of classes
    #print(preds)
    return preds

if __name__ == '__main__':
    fp = "../testkode/data17/"
    train = fp + "train/anger-ratings-0to1.train.txt"
    test = fp + "test/anger-ratings-0to1.test.target.txt"
    pred = fp + "test/preds.txt"

    #train_emotion, train_tweets, train_labels, train_ids = readTweetsOfficial(test)

    #print(train_tweets)

    train_features, train_labels, test_features, test_labels = extractFeatures(train, test)
    model = model_train(train_features, train_labels)
    predictions = predict(model, test_features)
    #print(predictions)

    printPredsToFile(test, pred, predictions)
    eval(test, pred)