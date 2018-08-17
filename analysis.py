# -*- coding: utf-8 -*
'''
Created on 2018年8月14日

@author: ls
'''
from utils import *
import pdb
import matplotlib.pyplot as plt

def countIntersectionWordsInDocs(docs, intersection_set):
    wordCountDict = {w: 0 for w in intersection_set}
    
    for doc in docs:
        splitted_doc = doc.split(" ")
        for word in splitted_doc:
            if word in wordCountDict:
                wordCountDict[word] += 1
    
    return wordCountDict




if __name__ == '__main__':
    debug = 4
    if debug == 1:
        train_path = 'train_set.csv'
        test_path = 'test_set.csv'
    elif debug == 0:
        train_path = 'train_set_sample.csv'
        test_path = 'test_set_sample.csv'
    elif debug == 3:
        train_path = r'D:\VM_Share\work\codeSpace\nlp\daguan\new_data\train_set.csv'
        test_path = r'D:\VM_Share\work\codeSpace\nlp\daguan\new_data\test_set.csv'
    else:
        train_path = r'D:\VM_Share\data\home\new_data\train_set.csv'
        test_path = r'D:\VM_Share\data\home\new_data\test_set.csv'
    
    #打印训练集与测试集文本最长、最短词数，并绘制训练集与测试集文本词个数分布
#     analyseLength(train_path, test_path)
#     for len_ in range(17, 40, 3):
#         getUniqWord(train_path, test_path, len_)
#         print("---------------------------------------------------")
    
    pdb.set_trace()
    train_docs, train_y = extract_docs(train_path)
    test_docs = extract_docs(test_path, isTrain = False)
    #     pdb.set_trace()
    
    train_words = extract_word_in_docs(train_docs)
    train_words_set = set(train_words)
    train_class_set = set(train_y)
    
    print("train_set has %d words in total, and has %d uniq words" % (len(train_words), len(train_words_set)))
    print("train_set has %s class" % len(train_class_set)); del train_class_set
    
    test_words = extract_word_in_docs(test_docs)
    test_words_set = set(test_words)
    
    print("test has %d words in total, and has %d uniq words" % (len(test_words), len(test_words_set)))
    
    del train_words, test_words
    isTestHasDifferentWordsInTrain(train_words_set, test_words_set)
    
    intersection_set = train_words_set.intersection(test_words_set)
    print("train and test has %d same words in total" % len(intersection_set))
    
#     trainWordsIntersectToTestCountDict = countIntersectionWordsInDocs(train_docs, intersection_set); del train_docs
#     print("intersection words between train and test has %d total words in train" % sum([trainWordsIntersectToTestCountDict[w] for w in trainWordsIntersectToTestCountDict]))
#     testWordsIntersectToTestCountDict = countIntersectionWordsInDocs(test_docs, intersection_set)
#     print("intersection words between train and test has %d total words in test" % sum([testWordsIntersectToTestCountDict[w] for w in testWordsIntersectToTestCountDict]))

    shortTextsWordsSet = keepUniqWordsOfShortTexts(train_path, test_path)
    
    pdb.set_trace()
    intersectionWordsBetweenShortTestAndIntersectionSet = intersection_set.intersection(shortTextsWordsSet)
    print("short texts has %d uniq words in total" % len(shortTextsWordsSet))
    print("intersection words between short text and intersection set has %d words in total." % len(intersectionWordsBetweenShortTestAndIntersectionSet))


    
    
    
    