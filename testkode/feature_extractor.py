from reader import *
from writer import *
from config import *
from sklearn.feature_extraction.text import TfidfVectorizer
import enchant
#from emoji import UNICODE_EMOJI
import emoji as Emoji

neg_emoji_list = [':frowning_face:', ':slightly_frowning_face:', ':face_with_steam_from_nose:',':crying_face:',
                  ':loudly_crying_face:',':weary_face:',':pouting_face:',':angry_face:', ':fearful_face:', ':grimacing_face:']
pos_emoji_list = [':grinning_face:', ':grinning_face_with_smiling_eyes:', ':face_with_tears_of_joy:', ':grinning_face_with_sweat:',
                  ':smiling_face_with_sunglasses:', ':smiling_face_with_heart_eyes:', ':hugging_face:', ':smiling_face:',
                  ':face_blowing_a_kiss:', ':winking_face:'] 

import sys
sys.path.append("../")
        
def featureMerger(tweets, exclam=False, hashtag=False, spelling=False, neg_emoji=False, pos_emoji=False, emoji=False):
    # Width defines the amount of columns in the custom feature matrix
    width = exclam + spelling + emoji + hashtag + neg_emoji + pos_emoji
    if (width==True): # If only 1 feature is set, manually set the width to 1
        width = 1
    feat_list = np.zeros((len(tweets), width))

    d = enchant.Dict('en_US')
    error_cntr = 0
    detected = False
    pos_detected = False
    neg_detected = False

    for i, tweet in enumerate(tweets):
        pos = width # To keep track of the current feature column to use, width is introduced
        words = tweet.split(" ")

        # Check whether or not the exclamation feature is set and do the check
        if ((exclam) and '!' in tweet):
            feat_list[i][width-pos] = 1
        # Keep track of the current feature column to use
        if (exclam):
            pos -= 1

        # Check whether or not the hashtag feature is set and do the check
        if ((hashtag) and '#' in tweet):
            feat_list[i][width-pos] = 1
        # Keep track of the current feature column to use
        if(hashtag):
            pos -= 1

        # Begin the word and character checking
        for word in words:

            # Check whether or not the word starts with a mention or hashtag, and sort them out of the spellchecking
            if ((word.startswith(('#', '@')) or (word == '')) and (spelling)):
                continue
            else:
                #Check the spelling and increase error count
                if ((spelling) and not(d.check(word))):
                    error_cntr+=1
            
            # Check every word for individual characters
            for j in range(len(word)):
                # Check whether or not a character is an emoji and whether or not the flags are set
                if ((word[j] in Emoji.UNICODE_EMOJI) and ((not(detected) and (emoji)) or (neg_emoji) or (pos_emoji))):
                    #print(word[j])
                    # Flag set if emoji detected, will break the loop if only emoji feature is set
                    detected = True

                    # Demojize the emoji to make lookup possible
                    demojized = Emoji.demojize(word[j])
                    # Check whether or not the emoji is possitive or negative
                    if (not(neg_detected) and (demojized in neg_emoji_list)):
                        neg_detected = True
                    if (not(pos_detected) and (demojized in pos_emoji_list)):
                        pos_detected = True

        # Check whether or not the amount of spelling mistakes are higher than given threshold
        if ((error_cntr/(len(words)) > 0.6) and (spelling)):
            feat_list[i][width-pos] = 1
        # Keep track of the current feature column to use
        if(spelling):
            pos -= 1

        # Check whether or not a negative emoji was found in tweet        
        if ((neg_detected) and (neg_emoji)):
            feat_list[i][width-pos] = 1
        # Keep track of the current feature column to use
        if (neg_emoji):
            pos -= 1
        
        # Check whether or not a positive emoji was found in tweet        
        if ((pos_detected) and (pos_emoji)):
            feat_list[i][width-pos] = 1
        # Keep track of the current feature column to use
        if (pos_emoji):
            pos -= 1
            
        # Check whether or not an emoji was found in tweet (OBS. no need to keep track of feature column on last feature)
        if ((detected) and (emoji)):
            feat_list[i][width-pos] = 1

        # Reset flags for every tweet
        detected = False
        pos_detected = False
        neg_detected = False
    return feat_list


def extractFeatures(train, test, dev, task, max_features, ngram_range, stop_words='english', analyzer='word'):
    if task == 'class':
        test_tweets_list = []
        train_tweets_list = []

        dev_tweets_list = []
        dev_labels_list = []

        test_labels_list = []
        train_labels_list = []

        for file in test:
            test_tweets, test_emotion, test_labels, test_ids = readTweetsOfficial(file, task=task)
            test_tweets_list.extend(test_tweets)
            test_labels_list.extend(test_labels)

        for file in dev: 
            dev_tweets, dev_emotion, dev_labels, dev_ids = readTweetsOfficial(file, task=task)
            dev_tweets_list.extend(dev_tweets)
            dev_labels_list.extend(dev_labels)

        for file in train:
            train_tweets, train_emotion, train_labels, train_ids = readTweetsOfficial(file, task=task)
            train_tweets_list.extend(train_tweets)
            train_labels_list.extend(train_labels)

        train_features, test_features, vocab = featTransform(train_tweets_list, test_tweets_list, analyzer=analyzer, max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
        
        return train_features, train_labels_list, test_features, test_labels_list
    
    elif task == 'reg':
        test_tweets_list = []
        train_tweets_list = []

        dev_tweets_list = []
        dev_labels_list = []

        test_labels_list = []
        train_labels_list = []

        test_features_list = []
        train_features_list = []

        for file in test:
            test_tweets, test_emotion, test_labels, test_ids = readTweetsOfficial(file, task=task)
            test_tweets_list.append(test_tweets)
            test_labels_list.append(test_labels)

        for file in train:
            train_tweets, train_emotion, train_labels, train_ids = readTweetsOfficial(file, task=task)
            train_tweets_list.append(train_tweets)
            train_labels_list.append(train_labels)

        for file in dev: 
            dev_tweets, dev_emotion, dev_labels, dev_ids = readTweetsOfficial(file, task=task)
            dev_tweets_list.append(dev_tweets)
            dev_labels_list.append(dev_labels)
        
        for i in range(4):
            train_features, test_features, vocab = featTransform(train_tweets_list[i], test_tweets_list[i], analyzer=analyzer, max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
            test_features_list.append(test_features)
            train_features_list.append(train_features)

        return train_features_list, train_labels_list, test_features_list, test_labels_list


def featTransform(train_tweets, test_tweets, analyzer, max_features, ngram_range, stop_words):
    # max_features=100, ngram_range=(1, 4), stop_words='english'
    TfidfV = TfidfVectorizer(analyzer=analyzer, max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
    TfidfV.fit(train_tweets)
    # print(TfidfV.vocabulary_)            
    train_features = TfidfV.transform(train_tweets)
    test_features = TfidfV.transform(test_tweets)
    train_features = train_features.todense()
    test_features = test_features.todense()

    train_custom_feat = featureMerger(train_tweets, exclam=args.exclam, hashtag=args.hashtag, spelling=args.spelling, neg_emoji=args.neg_emoji, pos_emoji=args.pos_emoji, emoji=args.emoji)
    train_features = np.append(train_features, train_custom_feat, 1)

    test_custom_feat = featureMerger(test_tweets, exclam=args.exclam, hashtag=args.hashtag, spelling=args.spelling, neg_emoji=args.neg_emoji, pos_emoji=args.pos_emoji, emoji=args.emoji)
    test_features = np.append(test_features, test_custom_feat, 1)
    # print(train_features)
    # print(test_features)
    return train_features, test_features, TfidfV.vocabulary_
