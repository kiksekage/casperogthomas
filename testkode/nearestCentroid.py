import sys
sys.path.append("../")

from reader import *
from writer import *
from feature_extractor import *
from sklearn.neighbors import NearestCentroid

stop_words = [(x.strip()) for x in open('arabic-stop-words.txt','r').read().split('\n')]

def model_train(train_features, train_emotions):
    model = NearestCentroid()
    model.fit(train_features, train_emotions)
    return model


def predict(model, test_features):
    preds = model.predict(test_features)
    # print(preds)
    return preds


if __name__ == '__main__':
    pred = "../testkode/preds.txt"

    if sys.argv[1] == '17':
        fp = "../testkode/data17/"

        train = [fp + "train/anger-ratings-0to1.train.txt", fp + "train/fear-ratings-0to1.train.txt",
                 fp + "train/joy-ratings-0to1.train.txt", fp + "train/sadness-ratings-0to1.train.txt"]
        test = [fp + "test/anger-ratings-0to1.test.target.txt", fp + "test/fear-ratings-0to1.test.target.txt",
                fp + "test/joy-ratings-0to1.test.target.txt", fp + "test/sadness-ratings-0to1.test.target.txt"]
        dev = [fp + "dev/anger-ratings-0to1.dev.txt", fp + "dev/fear-ratings-0to1.dev.txt",
               fp + "dev/joy-ratings-0to1.dev.txt", fp + "dev/sadness-ratings-0to1.dev.txt"]

        train_features, train_emotions, test_features, test_emotions = extractFeatures(train, test, 'class')  # ændr test til dev

        model = model_train(train_features, train_emotions)
        predictions = predict(model, test_features)

        # print(predictions)
        # print(train_emotions)

        printPredsToFileClass(test, pred, predictions)  # ændr test til dev

    elif sys.argv[1] == '18':
        fp = "../testkode/data18/2018-EI-reg-En-train/"
        fp_test = "../testkode/data17/"

        train = [fp + "2018-EI-reg-En-anger-train.txt", fp + "2018-EI-reg-En-fear-train.txt",
                 fp + "2018-EI-reg-En-joy-train.txt", fp + "2018-EI-reg-En-sadness-train.txt"]
        test = [fp_test + "test/anger-ratings-0to1.test.target.txt", fp_test + "test/fear-ratings-0to1.test.target.txt",
                fp_test + "test/joy-ratings-0to1.test.target.txt", fp_test + "test/sadness-ratings-0to1.test.target.txt"]
        
        train_features, train_emotions, test_features, test_emotions = extractFeatures(train, test, 'class')  # ændr test til dev

        model = model_train(train_features, train_emotions)
        predictions = predict(model, test_features)

        # print(predictions)
        # print(train_emotions)

        printPredsToFileClass(test, pred, predictions)  # ændr test til dev

    elif sys.argv[1] == 'arabic':
        fp = "../testkode/data18/2018-EI-reg-Ar-train/"
        fp_dev = "../testkode/data18/2018-EI-reg-Ar-dev/"

        train = [fp + "2018-EI-reg-Ar-anger-train.txt", fp + "2018-EI-reg-Ar-fear-train.txt",
                 fp + "2018-EI-reg-Ar-joy-train.txt", fp + "2018-EI-reg-Ar-sadness-train.txt"]
        dev = [fp_dev + "2018-EI-reg-Ar-anger-dev.txt", fp_dev + "2018-EI-reg-Ar-fear-dev.txt",
                fp_dev + "2018-EI-reg-Ar-joy-dev.txt", fp_dev + "2018-EI-reg-Ar-sadness-dev.txt"]

        train_features, train_emotions, test_features, test_emotions = extractFeatures(train, test, 'class')  # ændr test til dev

        model = model_train(train_features, train_emotions)
        predictions = predict(model, test_features)

        # print(predictions)
        # print(train_emotions)

        printPredsToFileClass(dev, pred, predictions)  # ændr test til dev
