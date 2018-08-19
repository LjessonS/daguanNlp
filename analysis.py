# -*- coding: utf-8 -*
'''
Created on 2018年8月14日

@author: ls
'''
from utils import *
import pdb
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.feature_extraction import DictVectorizer
from collections import Counter, OrderedDict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def countIntersectionWordsInDocs(docs, intersection_set):
    wordCountDict = {w: 0 for w in intersection_set}
    
    for doc in docs:
        splitted_doc = doc.split(" ")
        for word in splitted_doc:
            if word in wordCountDict:
                wordCountDict[word] += 1
    
    return wordCountDict

def getTempExcludedSamples(train_path, length = 30):
    train_docs, _ = extract_docs(train_path)
    tempExcludedSamples = []
    for ind, doc in enumerate(train_docs):
        splitted_doc = doc.split(' ')
        if len(splitted_doc) <= length:
            tempExcludedSamples.append(ind)
    return tempExcludedSamples

def selectKBestWords(train_path, test_path, lengthOfShortTextToKeep = 30, numBestKwords = 30000):
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
    
    isTestHasDifferentWordsInTrain(train_words_set, test_words_set)
    
    intersection_set = train_words_set.intersection(test_words_set)
    print("train and test has %d same words in total" % len(intersection_set))
    
    length = lengthOfShortTextToKeep
    shortTextsWordsSet = keepUniqWordsOfShortTexts(train_path, test_path, length)
    
    intersectionWordsBetweenShortTestAndIntersectionSet = intersection_set.intersection(shortTextsWordsSet)
    print("short texts has %d uniq words in total" % len(shortTextsWordsSet))
    print("intersection words between short text and intersection set has %d words in total." % len(intersectionWordsBetweenShortTestAndIntersectionSet))

#     candidateFeatures = intersection_set - shortTextsWordsSet
    
    tempExcludedSamples = getTempExcludedSamples(train_path, length)
    train_docs_clone, train_y_clone = deepcopy(train_docs), deepcopy(train_y)
    [train_docs_clone.pop(ind) for ind in tempExcludedSamples]
    [train_y_clone.pop(ind) for ind in tempExcludedSamples]

    train_words_clone = extract_doc_terms_in_docs(train_docs_clone)
    wordsToRemove = train_words_set - (intersection_set | shortTextsWordsSet)
    corpusToSelectWords = []
    print("start to remove words------------------------")
    for doc_terms in train_words_clone:
        temp = []
        for term in doc_terms:
            if term in wordsToRemove:
                continue
            temp.append(term)
        corpusToSelectWords.append(temp)
#     [[train_words_clone[ind].remove(w) for ind in range(len(train_words_clone))] for w in train_words_set]
    print("finish word-removing process---------------")
    
    v = DictVectorizer()

    print("start to generate doc words count---------------")
    X = v.fit_transform(Counter(f) for f in corpusToSelectWords)
    
    print("start to run chi2 words selection---------------")
    
    X_new = SelectKBest(chi2, k=numBestKwords).fit(X, train_y_clone)
    kBestWordsIndx = sorted(range(len(X_new.scores_)), key = lambda k: X_new.scores_[k], reverse = True)
    kBestWords = [v.feature_names_[ind] for ind in kBestWordsIndx][:numBestKwords]
    
    lastWordsToKeep = set(kBestWords) | shortTextsWordsSet
    return lastWordsToKeep
#     with open('kBestWords.txt', 'w') as f:
#         f.write('\t'.join(kBestWords))

def extract_doc_terms_within_wordsSet(doc_terms, lastWordsToKeep):
    fined_doc_terms = []
    for d_t in doc_terms:
        temp = []
        for term in d_t:
            if term in lastWordsToKeep:
                temp.append(term)
        fined_doc_terms.append(temp)
    return fined_doc_terms

def generateFinedTrainAndTest(train_path, test_path, lengthOfShortTextToKeep = 30, numBestKwords = 30000):
    lastWordsToKeep = selectKBestWords(train_path, test_path, lengthOfShortTextToKeep = 30, numBestKwords = 30000)
    train_docs, train_y = extract_docs(train_path)
    train_doc_terms = extract_doc_terms_in_docs(train_docs)
    
    finedTrain = extract_doc_terms_within_wordsSet(train_doc_terms, lastWordsToKeep)
    assert(len(train_docs) == len(finedTrain))
    
    test_docs = extract_docs(test_path, isTrain = False)
    test_doc_terms = extract_doc_terms_in_docs(test_docs)
    finedTest = extract_doc_terms_within_wordsSet(test_doc_terms, lastWordsToKeep)
    assert(len(test_docs) == len(finedTest))

    with open("finedTrain.csv", 'w') as f:
        for doc_terms, y in zip(train_doc_terms, train_y):
            f.write(' '.join(doc_terms)+','+y+'\n')
    with open("finedTest.csv", 'w') as f:
        for doc_terms in finedTest:
            f.write(' '.join(doc_terms)+'\n')
            
    
if __name__ == '__main__':
    debug = 1
    if debug == 0:
        train_path = 'train_set_sample.csv'
        test_path = 'test_set_sample.csv'
    if debug == 1:
        train_path = 'train_set.csv'
        test_path = 'test_set.csv'
    if debug == 2:
        train_path = r'D:\VM_Share\data\home\new_data\train_set.csv'
        test_path = r'D:\VM_Share\data\home\new_data\test_set.csv'
    if debug == 3:
        train_path = r'D:\VM_Share\work\codeSpace\nlp\daguan\new_data\train_set.csv'
        test_path = r'D:\VM_Share\work\codeSpace\nlp\daguan\new_data\test_set.csv'
    if debug == 4:
        train_path = r'D:\VM_Share\data\home\new_data\train_set_3000.csv'
        test_path = r'D:\VM_Share\data\home\new_data\test_set_3000.csv'
    
    #打印训练集与测试集文本最长、最短词数，并绘制训练集与测试集文本词个数分布
#     analyseLength(train_path, test_path)
#     for len_ in range(17, 40, 3):
#         getUniqWord(train_path, test_path, len_)
#         print("---------------------------------------------------")
    
    generateFinedTrainAndTest(train_path, test_path, numBestKwords = 300000)
    
    