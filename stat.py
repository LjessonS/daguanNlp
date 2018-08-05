# -*-coding:utf-8-*-
'''
Created on 2018骞�8鏈�4鏃� 涓嬪崍7:43:12

@author: liusheng1
'''
import pdb
from utils import extract_docs
import multiprocessing
from multiprocessing.pool import Pool
from multiprocessing import Value, Manager
import os

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

def getTextCount(voc, docs):
    doc_word_cnt = []
    for iii, doc in enumerate(docs):
        print("统计第 %d 篇文章的词频" % (iii+1))
        word_cnt = [0 for _ in voc]
        doc_split = doc.split(' ')
        occure = False
        for ind, v in enumerate(voc):
            for word in doc_split:
                if word == v:
                    word_cnt[ind] += 1
                    occure = True
            if occure:
                occure = False
                doc_split.remove(v)
        doc_word_cnt.append(word_cnt)
        
    return doc_word_cnt

def getDocCount(id_doc_voc):
    with open('stat.info', 'a+') as f:
        f.write("统计第 %d 篇文章的词频\n" % (id_doc_voc[0]+1))
    word_cnt = [0 for _ in id_doc_voc[2]]
    doc_split = id_doc_voc[1].split(' ')
    occure = False
    for ind, v in enumerate(id_doc_voc[2]):
        for word in doc_split:
            if word == v:
                word_cnt[ind] += 1
                occure = True
        if occure:
            occure = False
            doc_split.remove(v)
    return word_cnt

def setVoc(train_class_set, test_words_set):
    return sorted(train_class_set.union(test_words_set))

if __name__ == '__main__':
#     train_path = r'D:\VM_Share\data\home\new_data'
    if 0:
        train_path = 'train_set.csv'
        test_path = 'test_set.csv'
    else:
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
    
    voc = setVoc(train_class_set, test_words_set)
    
#     pdb.set_trace()
#     print('start train doc word count-----------------------')
#     
#     train_doc_word_cnt = getTextCount(voc, train_docs)
#     
#     print('start test doc word count-----------------------')
#     
#     test_doc_word_cnt = getTextCount(voc, test_docs)
#     
    
    
    #-------------------multiProcess------------------------
    if os.path.exists('stat.info'):
        os.remove('stat.info')
    print('start train doc word count-----------------------')
    train_id_doc_lst = [(ind, doc, voc) for ind, doc in enumerate(train_docs)]; del train_docs
    cpus = multiprocessing.cpu_count()
    pool = Pool(cpus-1)
    train_doc_word_cnt = pool.map(getDocCount, train_id_doc_lst)
    pool.close()
    pool.join()
    del train_id_doc_lst
    
    print('start test doc word count-----------------------')
    test_id_doc_lst = [(ind, doc, voc) for ind, doc in enumerate(test_docs)]; del test_docs
    cpus = multiprocessing.cpu_count()
    pool = Pool(cpus-1)
    test_doc_word_cnt = pool.map(getDocCount, test_id_doc_lst)
    pool.close()
    pool.join()
    del test_id_doc_lst
    
    print("Start output train doc word count-----------------------")
    with open('train_doc_word_cnt.txt', 'w') as f:
        f.write(' '.join(voc) + '\n')
        for doc_vec, label in zip(train_doc_word_cnt, train_y):
            f.write(' '.join(str(cnt) for cnt in doc_vec) + ' ' + label + '\n')
    
    print("output test doc word count-----------------------")
    
    with open('test_doc_word_cnt.txt', 'w') as f:
        f.write(' '.join(voc) + '\n')
        for doc_vec in test_doc_word_cnt:
            f.write(' '.join(str(cnt) for cnt in doc_vec) + '\n')
    
    
    