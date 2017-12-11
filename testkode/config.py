import argparse

fp = "../testkode/"

pred_dict = {0 : 'anger-pred.txt', 1 : 'fear-pred.txt', 2 : 'joy-pred.txt', 3 : 'sadness-pred.txt'}
pred_file = "../testkode/preds.txt"
pred_fold = fp + "preds/"


parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Train set', nargs='+')
parser.add_argument('--dev', help='Dev set', nargs='+')
parser.add_argument('--test', help='Test set', nargs='+')
parser.add_argument('--model', help='Model to use', type=str, choices=['perceptron', 'nearest_centroid', 'random_forest', 'random_forest_class'])
parser.add_argument('--exclam', help='Use exclamation feature', action='store_true')
parser.add_argument('--hashtag', help='Use hashtag feature', action='store_true')
parser.add_argument('--spelling', help='Use spelling mistake feature', action='store_true')
parser.add_argument('--neg_emoji', help='Use negative emoji feature', action='store_true')
parser.add_argument('--pos_emoji', help='Use positive emoji feature', action='store_true')
parser.add_argument('--emoji', help='Use emoji feature', action='store_true')
parser.add_argument('--max_features', help='Maximum features used', type=int, default=500)
parser.add_argument('--ngrams', help='ngram range used, if any', type=int, default=[1,1], nargs=2)

args = parser.parse_args()