import sys
from models import *
from reader import *
from writer import *
from feature_extractor import *
from config import *

#ar_stop_words = [(x.strip()) for x in open('arabic-stop-words.txt','r').read().split('\n')]

if args.model == 'random_forest_class':
    task = 'class'
else:
    task = 'reg'

ngram_range = (args.ngrams[0], args.ngrams[1])

def runner(model):
    if args.model == 'perceptron':
        train_features, train_labels, test_features, test_labels = extractFeatures(args.train, args.test, args.dev, task, args.max_features, ngram_range)
        model_object = train_perceptron(train_features, train_labels)
        preds = predictor(model_object, test_features)
        printPredsToFileClass(test_files, pred_file, preds)
        #dils = sorted(range(len(model_object.coef_[0])), key=lambda k: model_object.coef_[0][k])

    elif args.model == 'nearest_centroid':
        train_files, test_files = filepath_returner("class", language, year)
        train_features, train_labels, test_features, test_labels = extractFeatures(args.train, args.test, args.dev, task, args.max_features, ngram_range)
        model_object = train_nearest_centroid(train_features, train_labels)
        preds = predictor(model_object, test_features)
        printPredsToFileClass(test_files, pred_file, preds)

    elif args.model == 'random_forest':
        train_features, train_labels, test_features, test_labels = extractFeatures(args.train, args.test, args.dev, task, args.max_features, ngram_range)
        for i, emotion in enumerate(train_features):
            model_object = train_random_forest(emotion, train_labels[i])
            preds = predictor(model_object, test_features[i])    
            printPredsToFileReg(args.test[i], pred_fold + pred_dict[i], preds)
            #print((sorted(range(len(model_object.feature_importances_)), key=lambda k: model_object.feature_importances_[k])).index(501))
    elif args.model == 'random_forest_class':
        train_features, train_labels, test_features, test_labels = extractFeatures(args.train, args.test, args.dev, task, args.max_features, ngram_range)
        model_object = train_random_forest_class(train_features, train_labels)
        preds = predictor(model_object, test_features)
        printPredsToFileClass(test_files, pred_file, preds)


if __name__ == '__main__':
    #python3 sem_eval.py model language year
    runner(args.model)
