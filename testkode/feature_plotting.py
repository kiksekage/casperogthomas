import sys
import subprocess

max_f = 80
for i in range(1,10):
    subprocess.run(["python3", "nearestCentroid.py", "17", "{!s}".format(i*max_f)])
    f_score = subprocess.check_output(["sh", "eval.sh", "class"], encoding='UTF-8')
    print(f_score.split("\n")[8].split(" ")[3])