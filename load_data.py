# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 00:32:00 2020

@author: CM
"""

import jieba
import numpy as np
from nn.utils import load_txt
from nn.hyperparameters import Hyperparamters as hp

vocabulary = load_txt(hp.file_vocabulary)[:hp.sequence_length]
vocabulary_set = set(vocabulary)

def load_train_data(file_train_data=hp.file_train_data,
                    file_train_label=hp.file_train_label):
    train_data = [eval(l) for l in load_txt(file_train_data)]
    train_label = [int(l) for l in load_txt(file_train_label)]
    print('Load train data finished.')
    return np.array(train_data), np.array(train_label)


def load_test_data(file_test_data=hp.file_test_data,
                   file_test_label=hp.file_test_label):
    test_data = [eval(l) for l in load_txt(file_test_data)]
    test_label = [int(l) for l in load_txt(file_test_label)]
    print('Load test data finished.')
    return np.array(test_data), np.array(test_label)


def sentence2vector(sentence):
    words = [l for l in jieba.lcut(sentence) if l in vocabulary_set]
    index = [vocabulary.index(word) for word in words]
    arr = np.zeros(hp.sequence_length)
    for i in index:
        arr[i]=1
    return arr


if __name__ == "__main__":
    #
    train_data, train_label = load_train_data()
    print(train_data.shape, train_label.shape)
    #
    test_data, test_label = load_test_data()
    print(test_data.shape, test_label.shape)
    
    




    
    
    
    
    
    
    
