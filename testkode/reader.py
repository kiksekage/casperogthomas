import json
import numpy as np

def readTweetsOfficial(tweetfile, encoding='utf-8', task='class'):
    """
    Read tweets from official files
    :param topic: which topic to use, if topic="all", data for all topics is read
    :param encoding: which encoding to use, official stance data is windows-1252 encoded
    :param tweetcolumn: which column contains the tweets
    :return: list of tweets, list of emotions, list of labels
    """
    tweets = []
    emotions = []
    labels = []
    ids = []
    if task == 'reg':
        for line in open(tweetfile, encoding=encoding, mode='r'):
            if line.startswith('ID\t'):
                continue
            content = line.strip().split('\t')
            sent_id, sentence = content[0:2]
            label_temp = content[2:]

            emotions.append(label_temp[0])
            tweets.append(sentence)
            ids.append(sent_id)
            labels.append(float(label_temp[1]))
    elif task == 'oc':
        for line in open(tweetfile, encoding=encoding, mode='r'):
            if line.startswith('ID\t'):
                continue
            content = line.strip().split('\t')
            sent_id, sentence = content[0:2]
            label_temp = content[2:]

            emotions.append(label_temp)
            tweets.append(sentence)
            ids.append(sent_id)
            labels.append([float(x) for x in labels])
    else:
        for line in open(tweetfile, encoding=encoding, mode='r'):
            if line.startswith('ID\t'):
                continue
            content = line.strip().split('\t')
            sent_id, sentence = content[0:2]
            label_temp = content[2:]

            emotions.append('NONE')
            tweets.append(sentence)
            ids.append(sent_id)
            labels.append([float(x) for x in label_temp])

    return tweets,emotions,labels,ids