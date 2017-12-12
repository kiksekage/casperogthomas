import subprocess
import sys
import numpy as np
import sem_eval

def printPredsToFileClass(infile, outfile, res, infileenc="utf-8", skip=0):
    outf = open(outfile, 'w')
    
    if sem_eval.task == 'oc': #TODO
        with open(infile, encoding=infileenc, mode='r') as f:
            outf.write(f.readline())
            for i, line in enumerate(f):
                pred = res[i]
                outl = line.strip("\n").split("\t")
                for j, flag in enumerate(pred):
                    outl[j+2] = flag
                    outf.write("\t".join(outl) + '\n')
    else:
        outf = open(outfile, 'w', encoding=infileenc)
        with open(infile, encoding=infileenc, mode='r') as f:
            outf.write(f.readline())
            for i, line in enumerate(f):
                pred = [int(x) for x in res[i]]
                outl = line.strip("\n").split("\t")
                for j, flag in enumerate(pred):
                    outl[j+2] = str(flag)
                outf.write("\t".join(outl) + '\n')
    outf.close()

def printPredsToFileReg(infile, outfile, res, infileenc="utf-8"):
    outf = open(outfile, 'w', encoding=infileenc)
    with open(infile, encoding=infileenc, mode='r') as f:
        outf.write(f.readline())
        for i, line in f:        
            outl = line.strip("\n").split("\t")
            outl[3] = str(res[i])
            outf.write("\t".join(outl) + '\n')