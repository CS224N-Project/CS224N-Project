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

def pad_rationals(rawRationals, padLen, padToken='PAD'):
    rawLen = [len(x) for x in rawRationals]
    addPaddingLen = [padLen - x for x in rawLen]
    zipRatPadLen = zip(rawRationals, addPaddingLen)
    paddedRational = [raw + [padToken] * length for raw, length in zipRatPadLen]
    return paddedRational

def read_padded_rationals(path, padLen, padToken='PAD'):
    rawData = read_raw_rationals(path)
    return pad_rationals(rawData, padLen, padToken)

def _extract_rationals(preds, reviews):
    numReviews = preds.shape[0]
    numWords = preds.shape[1]
    rationals = list()
    for i in xrange(numReviews):
        subset = list()
        for j in xrange(numWords):
            if preds[i, j] == 1:
                subset.append(reviews[i][j])
        rationals.append(subset)
    return rationals

def extract_rationals(predsPath, reviewPath, padLen, padToken = 'PAD'):
    preds = read_tf_pred(predsPath)
    reviews = read_padded_rationals(reviewPath, padLen, padToken)
    rationals = _extract_rationals(preds, reviews)
    return rationals, reviews

pathPreds = 'trial.txt'
pathReview = 'annotations.txt'

preds = read_tf_pred(pathPreds)
maxPad = preds.shape[1]

rationals, reviews = extract_rationals(pathPreds, pathReview, maxPad)