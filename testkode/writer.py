import io
import subprocess
import sys
import numpy as np

def printPredsToFile(infile, outfile, res, infileenc="utf-8", skip=0):
    """
    Print predictions to file in SemEval format so the official eval script can be applied
    :param infile: official stance data for which predictions are made
    :param infileenc: encoding of the official stance data file
    :param outfile: file to print to
    :param res: python list of results. 0 for NONE predictions, 1 for AGAINST predictions, 2 for FAVOR
    :param skip: how many testing instances to skip from the beginning, useful if part of the file is used for dev instead of test
    """
    outf = open(outfile, 'w')
    cntr = 0
    for file in infile:
        for line in io.open(file, encoding=infileenc, mode='r'):
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

def printPredsToFileByID(infile, outfile, ids, res, infileenc='utf-8'):
    """
    Print predictions to file in SemEval format so the official eval script can be applied
    :param infile: official stance data for which predictions are made
    :param infileenc: encoding of the official stance data file
    :param outfile: file to print to
    :param res: python list of results. 0 for NONE predictions, 1 for AGAINST predictions, 2 for FAVOR
    :param skip: how many testing instances to skip from the beginning, useful if part of the file is used for dev instead of test
    """
    outf = open(outfile, 'w')
    for line in io.open(infile, encoding=infileenc, mode='r'): #for the unlabelled Trump dev file it's utf-8
        if line.strip("\n").startswith('ID\t'):
            outf.write(line.strip("\n"))
        else:
            outl = line.strip("\n").split("\t")
            realid = np.asarray(outl[0], dtype=np.float64)
            cnt = 0
            for tid in ids:
                if tid == realid:
                    idx = cnt
                    #print("found!")
                    break
                cnt += 1
            if res[idx] == 0:
                outl[3] = 'NONE'
            elif res[idx] == 1:
                outl[3] = 'AGAINST'
            elif res[idx] == 2:
                outl[3] = 'FAVOR'
            outf.write("\n" + "\t".join(outl))

    outf.close()()