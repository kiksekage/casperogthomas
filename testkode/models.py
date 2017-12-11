import sys
sys.path.append("../")

from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def train_perceptron(train_features, train_emotions):
    perceptron = Perceptron(max_iter=1000, tol=1e-3)
    perceptron.fit(train_features, train_emotions)
    return perceptron

def train_nearest_centroid(train_features, train_emotions):
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(train_features, train_emotions)
    return nearest_centroid

def train_random_forest(train_features, train_labels):
    random_forest = RandomForestRegressor(random_state=1, n_estimators=10)
    random_forest.fit(train_features, train_labels)
    #feat_importance = sorted(range(len(random_forest.feature_importances_)), key=lambda k: random_forest.feature_importances_[k])
    #print(feat_importance.index(len(feat_importance)-4))
    #print(feat_importance.index(len(feat_importance)-3))
    #print(feat_importance.index(len(feat_importance)-2))
    #print(feat_importance.index(len(feat_importance)-1))
    #print('\n')
    return random_forest

def train_random_forest_class(train_features, train_labels):
    random_forest = RandomForestClassifier()
    random_forest.fit(train_features, train_labels)
    return random_forest_class

def predictor(model, test_features):
    preds = model.predict(test_features)
    return preds

