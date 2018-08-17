# -*-coding:utf-8-*-
'''
Created on 2018年8月14日 下午1:40:57

@author: liusheng1
'''
from utils import extract_docs
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb


if __name__ == '__main__':
    debug = 2
    if debug == 1:
        train_path = 'train_set.csv'
        test_path = 'test_set.csv'
    elif debug == 0:
        train_path = 'train_set_sample.csv'
        test_path = 'test_set_sample.csv'
    else:
        train_path = r'D:\VM_Share\data\home\new_data\train_set.csv'
        test_path = r'D:\VM_Share\data\home\new_data\test_set.csv'

    _, train_y = extract_docs(train_path)
    
#     pdb.set_trace()
    y_true_lst = sorted(int(ele) for ele in train_y)
    plt.hist(y_true_lst)
#     plt.show()
    plt.savefig("classDist.png")