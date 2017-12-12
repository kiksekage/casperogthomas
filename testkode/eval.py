from scipy.stats import pearsonr
import numpy as np

emdict = {'anger' : 0, 'fear' : 1, 'joy' : 2, 'sadness' : 3}
revdict = {0 : 'anger', 1 : 'fear', 2 : 'joy', 3 : 'sadness'}
'''
hit = [0,0,0,0] #anger, fear, joy, sadness
emotion_guess = [0,0,0,0] #anger, fear, joy, sadness
emotion_count = [0,0,0,0] #anger, fear, joy, sadness
precision = [0,0,0,0]
recall = [0,0,0,0]
f_score = [0,0,0,0]
avg = 0

#print(emdict['anger'])

with open(infile, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip('\n').split('\t')
        emotion_count[emdict[line[2]]] += 1
        emotion_guess[emdict[line[3]]] += 1
        if emdict[line[2]] == emdict[line[3]]:
            hit[emdict[line[2]]] += 1

for i in range(4):
    precision[i] = hit[i]/emotion_guess[i]

for i in range(4):
    recall[i] = hit[i]/emotion_count[i]

for i in range(4):
    f_score[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])

for i in range(4):
    print("{!s} values:".format(revdict[i]))
    print("Precision : {!s}, recall : {!s}, f-score : {!s}".format(precision[i], recall[i], f_score[i]))

for score in f_score:
    avg += score
print("Macro f-score : {!s}".format(avg/len(f_score)))
'''
def calculate_reg(gold, preds):
    pearson = pearsonr(gold, preds)[0]

    gold_high = []
    pred_high = []

    for i, value in enumerate(gold):
        if value > 0.5:
            gold_high.append(gold[i])
            pred_high.append(preds[i])

    pearson_high = pearsonr(gold_high, pred_high)[0]

    return pearson, pearson_high

def calculate_class(preds, gold):
    import ipdb; ipdb.set_trace()
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
    avg_f_micro = sum(f_micro)/len(f_micro)

    p_macro = sum(correct_emotion_micro)/sum(assigned_emotion_micro)
    r_macro = sum(correct_emotion_micro)/sum(actual_emotion_micro)
    try:
        avg_f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    except ZeroDivisionError:
        avg_f_macro = 0
    
    return macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro
    

def evaluate(train_preds, train_labels, dev_preds, dev_labels, test_preds, test_labels):
    if len(train_preds) == 4:
        print('Sanity check:')
        for i, gold in enumerate(train_labels):
            pearson, pearson_high = calculate_reg(gold, train_preds[i])
            print("Pearson for train tweets, {0}: {1}".format(revdict[i], pearson))
            print("Pearson for > 0.5 train tweets, {0}: {1}".format(revdict[i], pearson_high))
        print()

        print('Pearson for dev set:')
        for i, gold in enumerate(dev_labels):
            pearson, pearson_high = calculate_reg(gold, dev_preds[i])
            print("Pearson for dev tweets, {0}: {1}".format(revdict[i], pearson))
            print("Pearson for > 0.5 dev tweets, {0}: {1}".format(revdict[i], pearson_high))
        print()

        print('Pearson for test set:')
        for i, gold in enumerate(test_labels):
            pearson, pearson_high = calculate_reg(gold, test_preds[i])
            print("Pearson for test tweets, {0}: {1}".format(revdict[i], pearson))
            print("Pearson for > 0.5 test tweets, {0}: {1}".format(revdict[i], pearson_high))
        print()
        
    else:
        print('Sanity check:')
        macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro = calculate_class(train_preds, train_labels)
        print("Global accuracy for train tweets: {0}".format(macro_accuracy))
        print("F-micro for emotion classes:")
        print(f_micro)
        print("and averaged: " + str(avg_f_micro))
        print()

        print('Accuracy for dev set:')
        macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro = calculate_class(dev_preds, dev_labels)
        print("Global accuracy for dev tweets: {0}".format(macro_accuracy))
        print("F-micro for emotion classes:")
        print(f_micro)
        print("and averaged: " + str(avg_f_micro))
        print()

        print('Accuracy for test set:')
        macro_accuracy, p_micro, r_micro, f_micro, avg_f_micro, p_macro, r_macro, avg_f_macro = calculate_class(test_preds, test_labels)
        print("Global accuracy for test tweets: {0}".format(macro_accuracy))
        print("F-micro for emotion classes:")
        print(f_micro)
        print("and averaged: " + str(avg_f_micro))
        print()