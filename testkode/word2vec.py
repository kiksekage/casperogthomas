import gensim

from sklearn.manifold import TSNE
from reader import *
from writer import *
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("../")

train_tweets, train_emotion, train_labels, train_ids = readTweetsOfficial("../testkode/data18/2018-EI-reg-En-train/2018-EI-reg-En-anger-train.txt")

#print(train_tweets)

model = gensim.models.Word2Vec(sentences=train_tweets, window=5, size=300, sg=1)

vocab = list(model.wv.vocab)
X = model[vocab]

tsne = TSNE(n_components=2)
my_tsne = tsne.fit_transform(X)

df = pd.concat([pd.DataFrame(my_tsne),
                pd.Series(vocab)],
               axis=1)

df.columns = ['x', 'y', 'word']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])
for i, txt in enumerate(df['word']):
    ax.annotate(txt, (df['x'].iloc[i], df['y'].iloc[i]))

plt.show()