import io
import subprocess
import sys
import numpy as np

def printPredsToFileClass(infile, outfile, res, infileenc="utf-8", skip=0):
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

def printPredsToFileReg(infile, outfile, res, infileenc="utf-8", skip=0):
    """
    Print predictions to file in SemEval format so the official eval script can be applied
    :param infile: official stance data for which predictions are made
    :param infileenc: encoding of the official stance data file
    :param outfile: file to print to
    :param res: python list of results. 0 for NONE predictions, 1 for AGAINST predictions, 2 for FAVOR
    :param skip: how many testing instances to skip from the beginning, useful if part of the file is used for dev instead of test
    """
    outf = open(outfile, 'w', encoding=infileenc)
    cntr = 0
    for line in io.open(infile, encoding=infileenc, mode='r'):
        if skip > 0:
            skip -= 1
        else:
            outl = line.strip("\n").split("\t")
            outl[3] = str(res[cntr])
            outf.write("\t".join(outl) + '\n')
            cntr += 1

    outf.close()