import gensim

from reader import *
from writer import *

import sys
sys.path.append("../")

train_tweets, train_emotion, train_labels, train_ids = readTweetsOfficial("../testkode/data18/2018-EI-reg-En-train/2018-EI-reg-En-anger-train.txt")

#print(train_tweets)

model = gensim.models.Word2Vec(sentences=train_tweets, window=10, size=300, sg=1)

print(model.wv.vocab)

#print(model.wv.most_similar(positive=["fuck"]))