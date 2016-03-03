#!/usr/bin/python

"""
run this script to obtain the features
./script/preprocess.py dataset/reviews.txt
"""

import numpy as np
import re
import sys


def eachWord(word):
    word = re.sub('[^a-zA-Z0-9\']', ' ', word).strip()
    if len(word) > 1 and not word.isdigit():
        return word.lower().split()


def readFile(filename):
    res = []
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.split()
            words, label = tokens[:-1], tokens[-1]
            words_list = []
            for word in words:
                if eachWord(word):
                    words_list += eachWord(word)
            res.append((words_list, label))
    return res


def buildFeature(documents):
    word_table = {}
    idx = 0
    for words, _ in documents:
        for word in words:
            if word not in word_table:
                word_table[word] = idx
                idx += 1
    X = np.empty((len(word_table), len(documents)))
    Y = np.empty(len(documents))
    for id_j, (words, label) in enumerate(documents):
        Y[id_j] = label
        for word in words:
            id_i = word_table[word]
            X[id_i, id_j] += 1

    return X, Y, word_table


def main():
    res = readFile(sys.argv[1])
    x, y, table = buildFeature(res)
    print 'x is the feature matrix # of feature * # of documents', x.shape
    print x
    print 'y is the label vector # of documents', y.shape
    print y
    print 'table is the word dict, each word point to a number in the feature'
    print table.items()[:10]

if __name__ == '__main__':
    main()
