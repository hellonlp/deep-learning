# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 02:31:02 2020

@author: CM
"""


import numpy as np

from nn.utils import load_model
from nn.networks import NeuralNetwork
from nn.hyperparameters import Hyperparamters as hp
from nn.load_data import load_test_data, sentence2vector

NN = NeuralNetwork()
w1, b1, w2, b2 = load_model(hp.file_load_model)


def get_labels(x):
    """
    x: a sample or some samples
    """
    output1, output2 = NN.forward(np.mat(x), w1, b1, w2, b2)
    return [1 if l > 0.5 else 0 for l in np.array(output2)]


def get_label_by_sentences(sentences):
    vectors = [sentence2vector(sentence)  for sentence in sentences]
    predict_label = get_labels(vectors)
    return ['垃圾' if l==1 else '非垃圾' for l in predict_label]


if __name__ == '__main__':
    ##
    # train_data,train_label = load_train_data()
    test_data, test_label = load_test_data()
    predict_labels = get_labels(test_data)
    accuracy = round(sum([1 if i == j else 0 for i, j in zip(test_label, predict_labels)]) / len(test_label), 4)
    print('accuracy:',accuracy)
    #
    ## 
    sentences = ['太原实施高毒农药定点经营、实名购买制度太原新闻网那个讯  昨日全市创建国家食品安全城市推进会发布消息，今年是我市创建国家食品安全示范城市的“攻坚”年，我市要全面提升食品安全保障水平，禁止剧毒高毒农药用于国家规定的农作物，维护群众舌尖上的安全。    今年，我市要实施高毒农药定点经营、实名购买制度，禁止剧毒高毒农药用于国家规定的农作物。加大科学种养技术培训力度，指导农户依法科学使用农药、兽药、化肥、饲料和饲料添加剂，严禁违规使用“瘦肉精”、孔雀石绿、硝基呋喃等禁用物质。实施餐饮业食品安全提升工程，推广明厨亮灶。开展创建放心菜、放心肉超市活动。探索推广食品“三小”行业（小作坊、小餐饮、小摊贩）集中规范管理，打造一批“三小”集中经营示范街。    目前我市已打造出杏花岭区万达广场、尖草坪区文兴路、迎泽区食品街等食品安全示范街和示范店。我市将继续丰富创建国家食品安全示范城市的内涵，以优异的成绩迎接省级食品安全城市验收评价，举全市之力如期实现创建国家食品安全示范城市目标。3月23日起，省食安办要对我市创建省级食品安全示范城市和省级食品安全示范县进行验收评价，目前已全面进入迎检阶段。',
                 '武汉市迎来了军运会，在全世界都是很有名气的',
                 '江　东　区　找　小　姐　江　东　区　找　姐　全　套　保　健　按　摩江　东　区　找　小　姐　全　套',
                 '南京所有公办、民办小学推行延时照顾服务核心阅读 小学离校时间早，难倒不少双职工家庭。',
                 '武汉大学和北京大学',
                 '停车难！杭州这些医院附近 今年都有新的停车场要开放或者新建主城区停车哪里最难？医院是个重灾区 ']
    print(get_label_by_sentences(sentences))
    vectors = [sentence2vector(sentence)  for sentence in sentences]
    
    
    
    
    
    
    
    
    
    
