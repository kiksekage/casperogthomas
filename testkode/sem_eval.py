import sys
from models import *
from file_variables import *
from reader import *
from writer import *
from feature_extractor import *

def runner(model, language, year):
    if model == 'perceptron':
        train_files, test_files = filepath_returner("class", language, year)
        train_features, train_emotions, test_features, test_emotions = extractFeatures(train_files, test_files, 'class')
        model_object = train_perceptron(train_features, train_emotions)
        preds = predictor(model_object, test_features)
        printPredsToFileClass(test_files, pred_file, preds)

    elif model == 'nearest_centroid':
        train_files, test_files = filepath_returner("class", language, year)
        train_features, train_emotions, test_features, test_emotions = extractFeatures(train_files, test_files, 'class')
        model_object = train_nearest_centroid(train_features, train_emotions)
        preds = predictor(model_object, test_features)
        printPredsToFileClass(test_files, pred_file, preds)
    
    elif model == 'random_forest':
        train_files, test_files = filepath_returner("reg", language, year)
        train_features, train_labels, test_features, test_labels = extractFeatures(train_files, test_files, 'reg')
        for i, emotion in enumerate(train_features):
            model_object = train_random_forest(emotion, train_labels[i])
            preds = predictor(model, test_features[i])    
            printPredsToFileReg(test_files[i], pred_fold + pred_dict[i], preds) 

if __name__ == '__main__':
    #python3 sem_eval.py model language year
    runner(sys.argv[1], sys.argv[2], sys.argv[3])
