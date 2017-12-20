from scipy.stats import pearsonr
import numpy as np

emdict = {'anger' : 0, 'fear' : 1, 'joy' : 2, 'sadness' : 3}
revdict = {0 : 'anger', 1 : 'fear', 2 : 'joy', 3 : 'sadness'}

def calculate_reg(gold, preds):
    pearson = pearsonr(gold, preds)[0]

    gold_high = []
    pred_high = []

    for i, value in enumerate(gold):
        if value > 0.5:
            gold_high.append(gold[i])
            pred_high.append(preds[i])

    pearson_high = pearsonr(gold_high, pred_high)[0]

    return np.round(pearson, decimals=3), np.round(pearson_high, decimals=3)

def calculate_class(preds, gold):
    for i, augmented in enumerate(gold):
        if sum(augmented) < 0:
            gold = np.delete(gold, i, axis=0)
            preds = np.delete(preds, i, axis=0)
    micro_accuracy = []

    actual_emotion_micro = [0]*12
    correct_emotion_micro = [0]*12
    assigned_emotion_micro = [0]*12

    p_micro = [0]*12
    r_micro = [0]*12
    f_micro = [0]*12
    avg_f_micro = 0

    p_macro = 0
    r_macro = 0
    avg_f_macro = 0


    for i, labels in enumerate(gold):
        
        #Convert to class representation
        gold_labels = np.where(labels == 1)[0]
        pred_labels = np.where(preds[i] == 1)[0]

        #Neutral emotions, all 11 emotions 0
        if len(gold_labels) == 0:
            gold_labels = np.append(gold_labels, 11)
        if len(pred_labels) == 0:
            pred_labels = np.append(pred_labels, 11)
        
        for value_gold in gold_labels:
            actual_emotion_micro[value_gold] += 1
            if value_gold in pred_labels:
                correct_emotion_micro[value_gold] += 1

        for value_pred in pred_labels:
            assigned_emotion_micro[value_pred] += 1
        
        intersection = len(np.intersect1d(gold_labels, pred_labels))
        union = len(np.union1d(gold_labels, pred_labels))
        micro_accuracy.append(intersection/union)

    macro_accuracy = sum(micro_accuracy)/len(gold)

    for i in range(12):
        try:
            p_micro[i] = correct_emotion_micro[i]/assigned_emotion_micro[i]
        except ZeroDivisionError:
            p_micro[i] = 0
        
        try:
            r_micro[i] = correct_emotion_micro[i]/actual_emotion_micro[i]
        except ZeroDivisionError:
            r_micro[i] = 0

        try:
            f_micro[i] = 2*p_micro[i]*r_micro[i]/(p_micro[i]+r_micro[i])
        except ZeroDivisionError:
            f_micro[i] = 0
    
    p_micro = np.round(p_micro, decimals=3)
    r_micro = np.round(r_micro, decimals=3)
    f_micro = np.round(f_micro, decimals=3)

    avg_f_micro = sum(f_micro)/len(f_micro)

    p_macro = sum(correct_emotion_micro)/sum(assigned_emotion_micro)
    r_macro = sum(correct_emotion_micro)/sum(actual_emotion_micro)
    try:
        avg_f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    except ZeroDivisionError:
        avg_f_macro = 0
    
    avg_f_macro = round(avg_f_macro, 3)
    avg_f_micro = round(avg_f_micro, 3)
    
    return macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro
    

def evaluate(train_preds, train_labels, dev_preds, dev_labels, test_preds, test_labels):
    helper_string = ''
    if len(train_preds) == 4:
        helper_string += ('Sanity check:\n')
        pearson_avg_train = []
        for i, gold in enumerate(train_labels):
            pearson, pearson_high = calculate_reg(gold, train_preds[i])
            helper_string += ("Pearson for train tweets, {0}: {1}\n".format(revdict[i], pearson))
            helper_string += ("Pearson for > 0.5 train tweets, {0}: {1}\n".format(revdict[i], pearson_high))
            pearson_avg_train.append(pearson)
        helper_string += ("Average Pearson for train tweets: {0:.3f}\n".format(sum(pearson_avg_train)/len(pearson_avg_train)))

        helper_string += ('Pearson for dev set:\n')
        pearson_avg_dev = []
        for i, gold in enumerate(dev_labels):
            pearson, pearson_high = calculate_reg(gold, dev_preds[i])
            helper_string += ("Pearson for dev tweets, {0}: {1}\n".format(revdict[i], pearson))
            helper_string += ("Pearson for > 0.5 dev tweets, {0}: {1}\n".format(revdict[i], pearson_high))
            pearson_avg_dev.append(pearson)
        helper_string += ("Average Pearson for dev tweets: {0:.3f}\n".format(sum(pearson_avg_dev)/len(pearson_avg_dev)))

        helper_string += ('Pearson for test set:\n')
        pearson_avg_test = []
        for i, gold in enumerate(test_labels):
            pearson, pearson_high = calculate_reg(gold, test_preds[i])
            helper_string += ("Pearson for test tweets, {0}: {1}\n".format(revdict[i], pearson))
            helper_string += ("Pearson for > 0.5 test tweets, {0}: {1}\n".format(revdict[i], pearson_high))
            pearson_avg_test.append(pearson)
        helper_string += ("Average Pearson for test tweets: {0:.3f}\n".format(sum(pearson_avg_test)/len(pearson_avg_test)))
        
    else:
        helper_string += ('Sanity check:\n')
        macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro = calculate_class(train_preds, train_labels)
        helper_string += ("Global accuracy for train tweets: {0:.3f}\n".format(macro_accuracy))
        helper_string += ("F-micro for emotion classes:\n")
        helper_string += str(f_micro)
        helper_string += '\n'
        helper_string += ("and averaged: " + str(avg_f_micro)+'\n')

        helper_string += ('Accuracy for dev set:\n')
        macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro = calculate_class(dev_preds, dev_labels)
        helper_string += ("Global accuracy for dev tweets: {0:.3f}\n".format(macro_accuracy))
        helper_string += ("F-micro for emotion classes:\n")
        helper_string += str(f_micro)
        helper_string += '\n'
        helper_string += ("and averaged: " + str(avg_f_micro)+'\n')

        helper_string += ('Accuracy for test set:\n')
        macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro = calculate_class(test_preds, test_labels)
        helper_string += ("Global accuracy for test tweets: {0:.3f}\n".format(macro_accuracy))
        helper_string += ("F-micro for emotion classes:\n")
        helper_string += str(f_micro)
        helper_string += '\n'
        helper_string += ("and averaged: " + str(avg_f_micro)+'\n')
    return helper_string