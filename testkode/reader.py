import io
import json
import numpy as np

def readTweetsOfficial(tweetfile, encoding='utf-8', tweetcolumn=2, task='class'):
    """
    Read tweets from official files
    :param topic: which topic to use, if topic="all", data for all topics is read
    :param encoding: which encoding to use, official stance data is windows-1252 encoded
    :param tweetcolumn: which column contains the tweets
    :return: list of tweets, list of targets, list of labels
    """
    tweets = []
    targets = []
    labels = []
    ids = []
    if task == 'reg':
        for line in io.open(tweetfile, encoding=encoding, mode='r'):
            if line.startswith('ID\t'):
                continue
            targets.append(line.split("\t")[tweetcolumn])
            tweets.append(line.split("\t")[tweetcolumn-1])
            lid = line.split("\t")[0]
            v = []
            v.append(lid)
            ids.append(v)
            if tweetcolumn > 1:
                labels.append(float(line.split("\t")[tweetcolumn+1].strip("\n")))
            else:
                labels.append("NONE")
    else:
        for line in io.open(tweetfile, encoding=encoding, mode='r'):
            if line.startswith('ID\t'):
                continue
            targets.append(line.split("\t")[tweetcolumn])
            tweets.append(line.split("\t")[tweetcolumn-1])
            lid = line.split("\t")[0]
            v = []
            v.append(lid)
            ids.append(v)
            if tweetcolumn > 1:
                labels.append(line.split("\t")[tweetcolumn+1].strip("\n"))
            else:
                labels.append("NONE")
    

    return tweets,targets,labels,ids