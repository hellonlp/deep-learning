# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 00:29:50 2020

@author: CM
"""

import time
import numpy as np
import matplotlib.pyplot as plot


# 当前时间正常格式
def time_now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def cut_list(data, size):
    """
    data: a list
    size: the size of cut
    """
    return [data[i * size:min((i + 1) * size, len(data))] for i in range(int(len(data) - 1) // size + 1)]


def load_txt(file):
    """
    load a txt
    """
    with  open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
    return lines


def save_txt(file, lines):
    lines = [l + '\n' for l in lines]
    with  open(file, 'w+', encoding='utf-8') as fp:  # a+添加
        fp.writelines(lines)


def shuffle_two(a1, a2):
    """
    随机打乱a1和a2两个列表
    """
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    return [a1[l] for l in ran], [a2[l] for l in ran]


def save_model(w1, b1, w2, b2, file_save_model):
    np.savez(file_save_model, w1, b1, w2, b2)
    print('Save model finished!')


def load_model(file_load_model):
    model = np.load(file_load_model)
    return model['arr_0'], model['arr_1'], model['arr_2'], model['arr_3']


def plot_loss(x,y,x2,y2):
    plot.plot(x,y,linewidth=1,color='b', label='Train loss')
    plot.plot(x2,y2,linewidth=1,color='r', label='Test loss')
    plot.title("Loss Graph",fontsize=15)
    plot.xlabel("Step",fontsize=10)
    plot.ylabel("Loss",fontsize=10)
    plot.tick_params(axis='both',labelsize=10)   
    plot.legend(loc='upper right')
    #plot.show()
    plot.savefig('image/loss.png',dpi=2000)
    plot.close()


def plot_accuracy(x,y,x2,y2):
    plot.plot(x,y,linewidth=1,color='b', label='Train accuracy')#,marker='*'
    plot.plot(x2,y2,linewidth=1,color='r', label='Test accuracy')
    #plot.plot(x,y,'ob', linewidth=1,label='Train accuracy',marker='.')#,marker='*'
    #plot.plot(x2,y2,'or',linewidth=1, label='Test accuracy',marker='.')    
    plot.title("Accuracy Graph",fontsize=15)
    plot.xlabel("Step",fontsize=10)
    plot.ylabel("Accuracy",fontsize=10)
    plot.tick_params(axis='both',labelsize=10)
    plot.legend(loc='lower right')
    #plot.show()
    plot.savefig('image/accuracy.png',dpi=2000)
    plot.close()
    
    
    
if __name__ == "__main__":
    print(time_now_string())
