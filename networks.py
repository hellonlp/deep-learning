# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 01:06:44 2020

@author: CM
"""

import numpy as np
from nn.hyperparameters import Hyperparamters as hp
from nn.modules import sparse_softmax_cross_entropy_with_logits
from nn.modules import full_connection_sigmoid, error_sample,full_connection_tanh
from nn.modules import back_propagation_cross_entropy_batch,back_propagation_cross_entropy_batch_2
from nn.modules import back_propagation_quadratic_batch,back_propagation_quadratic_batch_2

class NeuralNetwork(object):
    def __init__(self):
        # Hidden Layer
        self.weight1_initial = np.random.randn(hp.W1_size[0], hp.W1_size[1])
        self.bias1_initial = np.zeros([1, hp.bais1_size])
        # Output Layer
        self.weight2_initial = np.random.randn(hp.W2_size[0], hp.W2_size[1])
        self.bias2_initial = np.zeros([1, hp.bais2_size])

    def forward(self, x, w1, b1, w2, b2):
        return full_connection_tanh(x, w1, b1), full_connection_sigmoid(output1, w2, b2)

    def backward(self, x, y, output1, output2, w1, b1, w2, b2, batch_size, lr=hp.lr):
        return back_propagation_cross_entropy_batch_2(w1, b1, w2, b2, error_sample(y, output2), x, output2, output1, lr,
                                                    batch_size)
    
    def loss(self,y,yp):
        return sparse_softmax_cross_entropy_with_logits(yp,y)

    def accuracy(self, y, yp):
        predict = [1 if l > 0.5 else 0 for l in np.array(yp)]
        return round(sum([1 if i == j else 0 for i, j in zip(y, predict)]) / len(predict), 4)


if __name__ == "__main__":
    print(1)
