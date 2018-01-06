import sys
import subprocess

max_f = range(500,30500,500)

s = ""
for i in max_f:
    kald = ["python3", "sem_eval.py",
            "--train", "data/2018-E-c-En-train.txt",
            "--dev", "data/2018-E-c-En-dev.txt",
            "--test", "data/2018-E-c-En-dev.txt",
            "--model", "random_forest_class", "--max_features", "{:d}".format(i), "--hashtag", "--exclam", "--spelling", "--neg_emoji", "--pos_emoji", "--emoji", "--ngrams", "1", "5"]
    pearson = subprocess.check_output(kald, encoding='UTF-8')
    s += 'Max features: {0}'.format(i) + " Accuracy: {:s}".format(pearson.split("\n")[7].split(" ")[5]) + " F-micro: {:s} {:s} \n".format(pearson.split("\n")[9], pearson.split("\n")[10])
    print("Done with {:d}".format(i))
with open('preds.txt', mode='w') as f:
    f.write(s)
