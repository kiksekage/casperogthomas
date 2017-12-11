import subprocess
import sys
import numpy as np

def printPredsToFileClass(infile, outfile, res, infileenc="utf-8", skip=0):
    outf = open(outfile, 'w')
    cntr = 0
    for file in infile:
        for line in open(file, encoding=infileenc, mode='r'):
            if skip > 0:
                skip -= 1
            else:
                outl = line.strip("\n").split("\t")
                if res[cntr] == 'anger':
                    outl[3] = 'anger'
                elif res[cntr] == 'joy':
                    outl[3] = 'joy'
                elif res[cntr] == 'sadness':
                    outl[3] = 'sadness'
                else:
                    outl[3] = 'fear'
                outf.write("\t".join(outl) + '\n')
                cntr += 1
    outf.close()

def printPredsToFileReg(infile, outfile, res, infileenc="utf-8"):
    outf = open(outfile, 'w', encoding=infileenc)
    cntr = 0
    with open(infile, encoding=infileenc, mode='r') as f:
        outf.write(f.readline())
        for line in f:        
            outl = line.strip("\n").split("\t")
            outl[3] = str(res[cntr])
            outf.write("\t".join(outl) + '\n')
            cntr += 1