import sys
import subprocess

max_f = range(500,30500,500)

s = ""
for i in max_f:
    kald = ["python3", "sem_eval.py",
            "--train", "data/2018-EI-reg-En-anger-train.txt", "data/2018-EI-reg-En-fear-train.txt", "data/2018-EI-reg-En-joy-train.txt", "data/2018-EI-reg-En-sadness-train.txt",
            "--dev", "data/2018-EI-reg-En-anger-dev.txt", "data/2018-EI-reg-En-fear-dev.txt", "data/2018-EI-reg-En-joy-dev.txt", "data/2018-EI-reg-En-sadness-dev.txt",
            "--test", "data/2018-EI-reg-En-anger-dev.txt", "data/2018-EI-reg-En-fear-dev.txt", "data/2018-EI-reg-En-joy-dev.txt", "data/2018-EI-reg-En-sadness-dev.txt",
            "--model", "random_forest", "--max_features", "{:d}".format(i), "--hashtag", "--exclam", "--spelling", "--neg_emoji", "--pos_emoji", "--emoji", "--ngrams", "1", "5"]
    pearson = subprocess.check_output(kald, encoding='UTF-8')
    s += 'Max features: {0}'.format(i) + " Avg. pearson: {:s} \n".format(pearson.split("\n")[19].split(" ")[5])
    print("Done with {:d}".format(i))
with open('preds.txt', mode='w') as f:
    f.write(s)
