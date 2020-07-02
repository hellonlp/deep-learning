# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:42:41 2020

@author: cm
"""

import os
pwd = os.path.dirname(os.path.abspath(__file__))


class Hyperparamters:
    # parameters
    lr = 0.01
    num_epochs = 50 
    sequence_length = 5000
    num_labels = 1
    batch_size = 256

    # shape of w1,b1,w2,b2
    W1_size = [sequence_length, sequence_length]
    bais1_size = sequence_length
    W2_size = [sequence_length, num_labels]
    bais2_size = num_labels

    # train data 
    file_train_data = os.path.join(pwd,'data/train_vectors.txt')
    file_train_label = os.path.join(pwd,'data/train_labels.txt')
    # test data 
    file_test_data = os.path.join(pwd,'data/test_vectors.txt')
    file_test_label = os.path.join(pwd,'data/test_labels.txt')
    
    # vocabulary file
    file_vocabulary = os.path.join(pwd,'data/vocabulary.txt')
    
    # model file
    file_save_model = os.path.join(pwd,'model/model.npz')
    file_load_model = os.path.join(pwd,'model/on_use/model_4.npz')
