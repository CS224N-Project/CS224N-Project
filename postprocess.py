import numpy as np

def read_tf_pred(path):
    f = open(path)
    lines = list()
    for line in f:
        splitLine = map(float, line.rstrip().split(' '))
        splitLine = map(int, splitLine)
        lines.append(splitLine)
    lines = np.array(lines, dtype = np.int32)
    return lines

def read_raw_rationals(path):
    f = open(path)
    lines = list()
    for line in f:
        splitLine = line.rstrip().split('\t')[1].split(' ')
        lines.append(splitLine)
    return lines


pathPreds = 'trial.txt'
pathRaw = 'annotations.txt'

preds = read_tf_pred(pathPreds)
rawSentence = read_raw_rationals(pathRaw)