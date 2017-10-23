import sys
sys.path.append("../")

from reader import *
from writer import *
from feature_extractor import *
from sklearn.model_selection import cross_validate 
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
    
    #selector = VarianceThreshold(0.2)
    #train_features = selector.fit_transform(train_features)
    #test_features = selector.fit_transform(test_features)


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
    pred_dict = {0 : 'anger-pred.txt', 1 : 'fear-pred.txt', 2 : 'joy-pred.txt', 3 : 'sadness-pred.txt'}
    pred_fold = 'preds/'

    if sys.argv[1] == '17':
        fp = "../testkode/data17/"

        train = [fp + "train/anger-ratings-0to1.train.txt", fp + "train/fear-ratings-0to1.train.txt",
                 fp + "train/joy-ratings-0to1.train.txt", fp + "train/sadness-ratings-0to1.train.txt"]
        test = [fp + "test/anger-ratings-0to1.test.target.txt", fp + "test/fear-ratings-0to1.test.target.txt",
                fp + "test/joy-ratings-0to1.test.target.txt", fp + "test/sadness-ratings-0to1.test.target.txt"]
        dev = [fp + "dev/anger-ratings-0to1.dev.txt", fp + "dev/fear-ratings-0to1.dev.txt",
               fp + "dev/joy-ratings-0to1.dev.txt", fp + "dev/sadness-ratings-0to1.dev.txt"]

        train_features, train_labels, test_features, test_labels = extractFeatures(train, test, 'reg', ngram_range=(1,1), max_features=10000)  # aendr test til dev

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

        train_features, train_labels, test_features, test_labels = extractFeatures(train, test, 'reg', ngram_range=(1,3), max_features=5000)  # aendr test til dev

        cv_results = []
        for i, emotion in enumerate(train_features):
            model = model_train(emotion, train_labels[i])
            predictions = predict(model, test_features[i])
       
            cv_results.append(cross_validate(model, emotion, train_labels[i], scoring='r2', cv=6, n_jobs=2))
            #print(predictions)
            #print(train_labels)

            printPredsToFileReg(test[i], pred_fold + pred_dict[i], predictions)  # aendr test til dev
        print(cv_results[0]['test_score'])
        print(cv_results[1]['test_score'])
        print(cv_results[2]['test_score'])
        print(cv_results[3]['test_score'])

    elif sys.argv[1] == "arabic":
        fp = "../testkode/data18/2018-EI-reg-Ar-train/"
        fp_dev = "../testkode/data18/2018-EI-reg-Ar-dev/"

        train = [fp + "2018-EI-reg-Ar-anger-train.txt", fp + "2018-EI-reg-Ar-fear-train.txt",
                 fp + "2018-EI-reg-Ar-joy-train.txt", fp + "2018-EI-reg-Ar-sadness-train.txt"]
        dev = [fp_dev + "2018-EI-reg-Ar-anger-dev.txt", fp_dev + "2018-EI-reg-Ar-fear-dev.txt",
                fp_dev + "2018-EI-reg-Ar-joy-dev.txt", fp_dev + "2018-EI-reg-Ar-sadness-dev.txt"]

        train_features, train_labels, test_features, test_labels = extractFeatures(train, dev, 'reg', ngram_range=(1,1), max_features=10000)  # aendr test til dev

        for i, emotion in enumerate(train_features):
            model = model_train(emotion, train_labels[i])
            predictions = predict(model, test_features[i])
       
            #print(predictions)
            #print(train_labels)

            printPredsToFileReg(dev[i], pred_fold + pred_dict[i], predictions)  # aendr test til dev
