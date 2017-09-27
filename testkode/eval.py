import io

emdict = {'anger' : 0, 'fear' : 1, 'joy' : 2, 'sadness' : 3}
revdict = {0 : 'anger', 1 : 'fear', 2 : 'joy', 3 : 'sadness'}

def eval(infile):
    hit = [0,0,0,0] #anger, fear, joy, sadness
    emotion_guess = [0,0,0,0] #anger, fear, joy, sadness
    emotion_count = [0,0,0,0] #anger, fear, joy, sadness
    precision = [0,0,0,0]
    recall = [0,0,0,0]
    f_score = [0,0,0,0]

    #print(emdict['anger'])

    with open(infile, 'r') as f:
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