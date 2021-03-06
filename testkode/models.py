import sys
sys.path.append("../")

from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

random_state = 1337

def train_perceptron(train_features, train_emotions):
    perceptron = Perceptron(max_iter=1000, tol=1e-3)
    perceptron.fit(train_features, train_emotions)
    return perceptron

def train_nearest_centroid(train_features, train_emotions):
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(train_features, train_emotions)
    return nearest_centroid

def train_random_forest(train_features, train_labels):
    random_forest = RandomForestRegressor(random_state=random_state, n_estimators=2000)
    random_forest.fit(train_features, train_labels)
    
    '''feat_importance = sorted(range(len(random_forest.feature_importances_)), key=lambda k: random_forest.feature_importances_[k])
    print("exclam: " + str(feat_importance.index(1000)))
    print("hashtag: " + str(feat_importance.index(1001)))
    print("spelling: " + str(feat_importance.index(1002)))
    print("neg_emoji: " + str(feat_importance.index(1003)))
    print("pos_emoji: " + str(feat_importance.index(1004)))
    print("emoji: " + str(feat_importance.index(1005)))

    print('\n')
    '''
    return random_forest

def train_random_forest_class(train_features, train_labels):
    random_forest_class = RandomForestClassifier(random_state=random_state, n_estimators=1000)
    random_forest_class.fit(train_features, train_labels)
    
    '''feat_importance = sorted(range(len(random_forest_class.feature_importances_)), key=lambda k: random_forest_class.feature_importances_[k])
    print("exclam " + str(feat_importance.index(1000)))
    print("hashtag " + str(feat_importance.index(1001)))
    print("spelling " + str(feat_importance.index(1002)))
    print("pos_emoji " + str(feat_importance.index(1003)))
    print("neg_emoji " + str(feat_importance.index(1004)))
    print("emoji " + str(feat_importance.index(1005)))'''
    return random_forest_class

def predictor(model, features):
    preds = model.predict(features)
    return preds

