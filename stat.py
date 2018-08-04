# -*-coding:utf-8-*-
'''
Created on 2018年8月4日 下午7:43:12

@author: liusheng1
'''
import pdb
from utils import extract_docs

def extract_word_in_docs(docs):
    words = []
    for doc in docs:
        wordLst = doc.split(' ')
        words += wordLst
        
    return words

def isTestHasDifferentWordsInTrain(train_words_set, test_words_set):
    cnt = 0
    for word in test_words_set:
        if word not in train_words_set:
            cnt += 1
#             print("test_set has %d words and word %s is not in train_set" % (cnt, word))
    print("test_set has %d words not in train_set" % cnt)

if __name__ == '__main__':
#     train_path = r'D:\VM_Share\data\home\new_data'
#     train_path = 'train_set.csv'
#     test_path = 'test_set.csv'
    train_path = 'train_set_sample.csv'
    test_path = 'test_set_sample.csv'
    
    train_docs, train_y = extract_docs(train_path)
    test_docs = extract_docs(test_path, isTrain = False)
#     pdb.set_trace()
    
    train_words = extract_word_in_docs(train_docs)
    train_words_set = set(train_words)
    train_class_set = set(train_y)
    
    print("train_set has %d words in total, and has %d uniq words" % (len(train_words), len(train_words_set)))
    print("train_set has %s class" % len(train_class_set))
    
    test_words = extract_word_in_docs(test_docs)
    test_words_set = set(test_words)
    
    print("test has %d words in total, and has %d uniq words" % (len(test_words), len(test_words_set)))

    isTestHasDifferentWordsInTrain(train_words_set, test_words_set)
    
    
    
    