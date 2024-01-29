# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 00:20:28 2020

@author: CM
"""


import numpy as np

from nn.networks import NeuralNetwork
from nn.hyperparameters import Hyperparamters as hp
from nn.load_data import load_train_data,load_test_data
from nn.utils import cut_list, time_now_string, save_model, shuffle_two
from nn.utils import plot_loss, plot_accuracy


NN = NeuralNetwork()


def train(x, y, x_test, y_test, batch_size=hp.batch_size):
    test_data, test_label = x_test, y_test
    w1, w2 = NN.weight1_initial, NN.weight2_initial
    b1, b2 = NN.bias1_initial, NN.bias2_initial
    global_steps,global_loss,global_accuracy,global_step = [],[],[],0
    global_loss_test,global_accuracy_test = [],[]
    for epoch in range(hp.num_epochs):#
        x_input,y_input = shuffle_two(x,y)
        x_bloc = cut_list(x_input, batch_size)        
        y_bloc = cut_list(y_input, batch_size)
        print('Number of batchs:', len(x_bloc))
        for i in range(0, len(x_bloc)):     
            # Forward
            output1, output2 = NN.forward(np.array(x_bloc[i]), w1, b1, w2, b2)
            # Backward
            w1, b1, w2, b2 = NN.backward(np.array(x_bloc[i]), np.array([y_bloc[i]]).T, output1, output2, w1, b1, w2, b2,
                                         hp.lr, batch_size)
            # Train loss and accuracy
            loss = NN.loss(np.array([y_bloc[i]]).T, output2, batch_size)
            accuracy = NN.accuracy(y_bloc[i], output2)
            global_loss.append(loss)
            global_accuracy.append(accuracy)
            global_step = global_step + 1
            global_steps.append(global_step)
            # Test loss and accuracy
            output1_test, output2_test = NN.forward(np.array(x_test), w1, b1, w2, b2)
            loss_test = NN.loss(np.array([y_test]).T, output2_test, len(y_test))
            accuracy_test = NN.accuracy(y_test, output2_test)
            global_loss_test.append(loss_test) 
            global_accuracy_test.append(accuracy_test)
            # Log
            if i % 5 == 0:
                print('\033[1;35m', 'Train data. Time now:%s. Epoch number:%s, has finished:%s. Loss:%s, Accuracy:%s \033[0m' % (
                    time_now_string(), epoch, "%.2f%%" % (100 * (i + 1) / len(x_bloc)), loss, accuracy))
                print('\033[1;35m', 'Test  data. Time now:%s. Epoch number:%s. Loss:%s, Accuracy:%s \033[0m' % (
                    time_now_string(), epoch, loss_test, accuracy_test))                
        # Save model
        save_model(w1, b1, w2, b2, 'model/model_%s.npz' % (epoch))
    # Plot
    plot_loss(global_steps,global_loss,global_steps,global_loss_test)
    plot_accuracy(global_steps,global_accuracy,global_steps,global_accuracy_test)
    print('Train Done!')
    return w1, b1, w2, b2


if __name__ == '__main__':
    # Load data
    train_data, train_label = load_train_data()
    test_data, test_label = load_test_data()
    # Training
    w1, w2, b1, b2 = train(train_data, train_label, test_data, test_label)

