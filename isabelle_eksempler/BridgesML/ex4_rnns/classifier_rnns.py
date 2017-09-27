#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'

import sys
sys.path.append("/Users/thomas/Skole/Dropbox/SKOLE/bachelorprojekt/isabelle eksempler/BridgesML")

from readwrite.reader import *
from readwrite.writer import *
import tensorflow as tf
from ex4_rnns.tfutil import BatchBucketSampler, LossHook, Trainer, load_model
from ex3_word2vec.tokenize_tweets import tokenise_tweets
import collections


def loadData(trainingdata, testdata):
    tweets_train, targets_train, labels_train, ids_train = readTweetsOfficial(trainingdata)
    tweets_test, targets_test, labels_test, ids_test = readTweetsOfficial(testdata)

    transf_labels_train = transform_labels(labels_train)
    transf_labels_test = transform_labels(labels_test)

    tweet_tokens_train = tokenise_tweets(tweets_train)
    target_tokens_train = tokenise_tweets(targets_train)

    tweet_tokens_test = tokenise_tweets(tweets_test)
    target_tokens_test = tokenise_tweets(targets_test)

    #w2vmodel = word2vec.Word2Vec.load("../out/skip_nostop_multi_100features_5minwords_5context")
    count, dictionary, reverse_dictionary = build_dataset([token for senttoks in tweet_tokens_train+target_tokens_train for token in senttoks])  #flatten tweets for vocab construction

    transformed_tweets_train = [transform_tweet_dict(dictionary, senttoks) for senttoks in target_tokens_train]
    transformed_targets_train = [transform_tweet_dict(dictionary, senttoks) for senttoks in target_tokens_train]

    transformed_tweets_test = [transform_tweet_dict(dictionary, senttoks) for senttoks in tweet_tokens_test]
    transformed_targets_test = [transform_tweet_dict(dictionary, senttoks) for senttoks in target_tokens_test]

    #X = w2vmodel.syn0
    #vocab_size = len(w2vmodel.vocab)

    vocab_size = len(dictionary)

    return transformed_tweets_train, transformed_targets_train, transf_labels_train, ids_train, transformed_tweets_test, transformed_targets_test, transf_labels_test, ids_test, vocab_size


def transform_labels(labels):
    labels_t = []
    for lab in labels:
        v = np.zeros(3)
        if lab == 'NONE':
            ix = 0
        elif lab == 'AGAINST':
            ix = 1
        elif lab == 'FAVOR':
            ix = 2
        v[ix] = 1
        labels_t.append(v)
    return labels_t


def build_dataset(words, vocabulary_size=5000000, min_count=5):
    """
    Build vocabulary, code based on tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    :param words: list of words in corpus
    :param vocabulary_size: max vocabulary size
    :param min_count: min count for words to be considered
    :return: counts, dictionary mapping words to indeces, reverse dictionary
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        if _ >= min_count:# or _ == -1:  # that's UNK only
            dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print("Final vocab size:", len(dictionary))
    return count, dictionary, reverse_dictionary


def transform_tweet_dict(dictionary, words, maxlen=20):
    """
    Transform list of tokens, add padding to maxlen
    :param dictionary: dict which maps tokens to integer indices
    :param words: list of tokens
    :param maxlen: maximum length
    :return: transformed tweet, as numpy array
    """
    data = list()
    for i in range(0, maxlen-1):  #range(0, len(words)-1):
        if i < len(words):
            word = words[i]
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
        else:
            index = 0
        data.append(index)
    return np.asarray(data)

def transform_tweet(w2vmodel, words, maxlen=20):
    """
    Transform list of tokens with word2vec model, add padding to maxlen
    :param w2vmodel: word2vec model
    :param words: list of tokens
    :param maxlen: maximum length
    :return: transformed tweet, as numpy array
    """
    data = list()
    for i in range(0, maxlen-1):  #range(0, len(words)-1):
        if i < len(words):
            word = words[i]
            if word in w2vmodel.vocab:
                index = w2vmodel.vocab[word].index
            else:
                index = w2vmodel.vocab["unk"].index
        else:
            index = w2vmodel.vocab["unk"].index
        data.append(index)
    return np.asarray(data)



def get_model_concat(batch_size, max_seq_length, input_size, hidden_size, softm_target_size, vocab_size):

    """
    LSTM over target and over tweet, concatenated
    """

    # batch_size x max_seq_length
    inputs_tweet = tf.placeholder(tf.int32, [batch_size, max_seq_length])
    inputs_target = tf.placeholder(tf.int32, [batch_size, max_seq_length])

    embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, input_size], -0.1, 0.1),  # input_size is embeddings size
                                   name="embedding_matrix", trainable=True)

    # batch_size x max_seq_length x input_size
    embedded_tweet = tf.nn.embedding_lookup(embedding_matrix, inputs_tweet)
    embedded_target = tf.nn.embedding_lookup(embedding_matrix, inputs_target)

    embedded_inputs_all = tf.concat(1, [embedded_tweet, embedded_target])  # concatenating the two embeddings

    # [batch_size x inputs_size] with max_seq_length elements
    # inputs_list[0]: batch_size x input[0] <-- word vector of the first word
    inputs_list = [tf.squeeze(x) for x in
                   tf.split(1, max_seq_length * 2, embedded_inputs_all)]

    with tf.variable_scope("RNN"):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
        # returning [batch_size, max_time, cell.output_size]
        outputs, states = tf.nn.rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=inputs_list)


    """dim1, dim2 = tf.unpack(tf.shape(outputs))  # [h_i], [h_i, c_i]
    slice_size = [dim1, 1]
    slice_begin = [0, dim2-1]
    outputs_fin = tf.squeeze(tf.slice(outputs, slice_begin, slice_size), [1])"""

    outputs_fin = outputs[-1]

    weight = tf.Variable(tf.truncated_normal([hidden_size, softm_target_size], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[softm_target_size]))

    prediction = tf.nn.softmax(tf.matmul(outputs_fin, weight) + bias)

    return prediction, [inputs_tweet, inputs_target]



def create_softmax_loss(scores, target_values):
    """

    :param scores: [batch_size, num_candidates] logit scores
    :param target_values: [batch_size, num_candidates] vector of 0/1 target values.
    :return: [batch_size] vector of losses (or single number of total loss).
    """
    return tf.nn.softmax_cross_entropy_with_logits(scores, target_values)


def test_trainer(tweets_tr, targets_tr, labels_tr, ids_tr, tweets_te, targets_te, labels_te, ids_te, vocab_size):

    # hyperparameters
    learning_rate = 0.0001
    batch_size = 98
    input_size = 100
    hidden_size = 60
    max_epochs = 21

    max_seq_length = len(tweets_tr[0])
    softm_target_size = 3

    model, placeholders = get_model_concat(batch_size, max_seq_length, input_size, hidden_size, softm_target_size, vocab_size)

    ids = tf.placeholder(tf.float32, [batch_size, 1], "ids")  #ids are so that the dev/test samples can be recovered later
    softm_targets = tf.placeholder(tf.float32, [batch_size, softm_target_size], "targets")

    loss = tf.nn.softmax_cross_entropy_with_logits(model, softm_targets)

    data = [np.asarray(tweets_tr), np.asarray(targets_tr), np.asarray(ids_tr), np.asarray(labels_tr)]

    optimizer = tf.train.AdamOptimizer(learning_rate)
    batcher = BatchBucketSampler(data, batch_size)

    placeholders += [ids]
    placeholders += [softm_targets]

    pad_nr = batch_size - (len(labels_te) % batch_size) + 1  # since train/test batches need to be the same size, add padding for test

    data_test = [np.lib.pad(np.asarray(tweets_te), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(targets_te), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(ids_te), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0)),
                 np.lib.pad(np.asarray(labels_te), ((0, pad_nr), (0, 0)), 'constant', constant_values=(0))
                 ]

    corpus_test_batch = BatchBucketSampler(data_test, batch_size)




    with tf.Session() as sess:

        trainer = Trainer(optimizer, max_epochs)

        trainer(batcher=batcher, placeholders=placeholders, loss=loss, model=model, session=sess)

        print("Applying to test data, getting predictions for NONE/AGAINST/FAVOR")

        predictions_detailed_all = []
        predictions_all = []
        ids_all = []

        total = 0
        correct = 0
        for values in corpus_test_batch:
            total += len(values[-1])
            feed_dict = {}
            for i in range(0, len(placeholders)):
                feed_dict[placeholders[i]] = values[i]
            truth = np.argmax(values[-1], 1)  # values[2] is a 3-length one-hot vector containing the labels. this is to transform those back into integers
            predictions = sess.run(tf.nn.softmax(model), feed_dict=feed_dict)
            predictions_detailed_all.extend(predictions)
            ids_all.extend(values[-2])
            predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1),
                                 feed_dict=feed_dict)
            predictions_all.extend(predicted)
            correct += sum(truth == predicted)

            print("Num testing samples " + str(total) +
                  "\tAcc " + str(float(correct) / total) +
                  "\tCorrect " + str(correct) + "\tTotal " + str(total))


    return predictions_all, predictions_detailed_all, ids_all



if __name__ == '__main__':

    np.random.seed(1337)
    tf.set_random_seed(1337)

    fp = "/Users/thomas/Skole/Dropbox/SKOLE/bachelorprojekt/isabelle eksempler/BridgesML/data/semeval/"
    train_path = fp + "semeval2016-task6-train+dev.txt"
    test_path = fp + "SemEval2016-Task6-subtaskB-testdata-gold.txt"
    pred_path = fp + "SemEval2016-Task6-subtaskB-testdata-pred.txt"

    tweets_tr, targets_tr, labels_tr, ids_tr, tweets_te, targets_te, labels_te, ids_te, vocab_size = loadData(train_path, test_path)
    predictions_all, predictions_detailed_all, ids_all = test_trainer(tweets_tr, targets_tr, labels_tr, ids_tr, tweets_te, targets_te, labels_te, ids_te, vocab_size)


    printPredsToFileByID(test_path, pred_path, ids_all, predictions_all)
    eval(test_path, pred_path)