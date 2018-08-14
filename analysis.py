# -*- coding: utf-8 -*
'''
Created on 2018年8月14日

@author: ls
'''
from utils import extract_docs
import pdb
import matplotlib.pyplot as plt

def print_longest_lowest(docs):
    longest = 0
    lowest = 99999
    for ind, doc in enumerate(docs):
        doc_len = len(doc.split(' '))
        if longest < doc_len:
            longest = doc_len
        if lowest > doc_len:
            lowest = doc_len
    
    print(longest, lowest)
   
def plot_length_dist(docs):
    length_lst = []
    for ind, doc in enumerate(docs):
        doc_len = len(doc.split(' '))
        length_lst.append(doc_len)
    
    length_lst = sorted(length_lst)
    plt.bar(range(len(length_lst)), length_lst, color='rgb')
    plt.show()
    
if __name__ == '__main__':
    debug = 3
    if debug == 1:
        train_path = 'train_set.csv'
        test_path = 'test_set.csv'
    elif debug == 0:
        train_path = 'train_set_sample.csv'
        test_path = 'test_set_sample.csv'
    elif debug == 3:
        train_path = r'D:\VM_Share\work\codeSpace\nlp\daguan\new_data\train_set.csv'
        test_path = r'D:\VM_Share\work\codeSpace\nlp\daguan\new_data\test_set.csv'
    
    train_docs, _ = extract_docs(train_path)
    
    print_longest_lowest(train_docs)
    plot_length_dist(train_docs)
    del train_docs
    
    test_docs = extract_docs(test_path, isTrain = False)
    print_longest_lowest(test_docs)
    plot_length_dist(test_docs)
    
    
    
    
    
    