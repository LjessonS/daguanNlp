# -*-coding:utf-8-*-
'''
Created on 2018骞�8鏈�4鏃� 涓嬪崍7:43:12

@author: liusheng1
'''
import pdb
from utils import extract_docs
import multiprocessing
from multiprocessing.pool import Pool
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

def getDocCount(doc):
    with open('stat.info', 'a+') as f:
        f.write("统计了一篇文章的词频\n")
    word_cnt_dict = {}
    for word in doc.split(' '):
        if word not in word_cnt_dict:
            word_cnt_dict[word] = 1
        else:
            word_cnt_dict[word] += 1
    return word_cnt_dict

def setVoc(train_class_set, test_words_set):
    return sorted(train_class_set.union(test_words_set))

if 1:
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
print("train_set has %s class" % len(train_class_set)); del train_class_set

test_words = extract_word_in_docs(test_docs)
test_words_set = set(test_words)

print("test has %d words in total, and has %d uniq words" % (len(test_words), len(test_words_set)))

isTestHasDifferentWordsInTrain(train_words_set, test_words_set)

voc = setVoc(train_words_set, test_words_set); del train_words_set, test_words_set

def getDocCountToOutput(doc_dict):
    line_dict = {v: 0 for v in voc}
    for v in doc_dict:
        line_dict[v] = doc_dict[v]
    del doc_dict
    return line_dict

if __name__ == '__main__':
#     pdb.set_trace()
#     print('start train doc word count-----------------------')
#     
#     train_doc_word_cnt = getTextCount(voc, train_docs)
#     
#     print('start test doc word count-----------------------')
#     
#     test_doc_word_cnt = getTextCount(voc, test_docs)
    
    
    #-------------------multiProcess------------------------
    if os.path.exists('stat.info'):
        os.remove('stat.info')
    print('start train doc word count-----------------------')
    cpus = multiprocessing.cpu_count()
    pool = Pool(cpus-1)
    train_doc_word_cnt = pool.map(getDocCount, train_docs)
    pool.close()
    pool.join()
    del train_docs
    
    print('start test doc word count-----------------------')
    cpus = multiprocessing.cpu_count()
    pool = Pool(cpus-1)
    test_doc_word_cnt = pool.map(getDocCount, test_docs)
    pool.close()
    pool.join()
    del test_docs
    
    batch_size = 16
    print("Start output train doc word count-----------------------")
    with open('train_doc_word_cnt.txt', 'w') as f:
        f.write(' '.join(voc) + ' label' + '\n')
        
        ind = 0
        fac, residue = len(train_doc_word_cnt) // batch_size, len(train_doc_word_cnt) - batch_size * len(train_doc_word_cnt) // batch_size
        for epoch in range(fac):
            if epoch != fac:
                bat_begin, bat_end = epoch * batch_size, (epoch + 1) * batch_size
            else:
                bat_begin, bat_end = epoch * fac, epoch * fac + residue
            
            cpus = multiprocessing.cpu_count()
            pool = Pool(8)
            train_doc_dict_lst = pool.map(getDocCountToOutput, train_doc_word_cnt[bat_begin:bat_end])
            pool.close()
            pool.join()
            
            for doc_line_dict, label in zip(train_doc_dict_lst, train_y[bat_begin:bat_end]):
                print('输出了 %d 行\n' % ind)
                ind += 1
                f.write(' '.join(str(doc_line_dict[v]) for v in voc) + ' ' + label + '\n')
                
            del train_doc_dict_lst
    del train_doc_word_cnt, train_y
    
    print("output test doc word count-----------------------")
    
    with open('test_doc_word_cnt.txt', 'w') as f:
        ind = 0
        f.write(' '.join(voc) + '\n')
        
        fac, residue = len(test_doc_word_cnt) // batch_size, len(test_doc_word_cnt) - batch_size * len(test_doc_word_cnt) // batch_size
        for epoch in range(fac):
            if epoch != fac:
                bat_begin, bat_end = epoch * batch_size, (epoch + 1) * batch_size
            else:
                bat_begin, bat_end = epoch * fac, epoch * fac + residue
            
            cpus = multiprocessing.cpu_count()
            pool = Pool(8)
            test_doc_dict_lst = pool.map(getDocCountToOutput, test_doc_word_cnt[bat_begin:bat_end])
            pool.close()
            pool.join()
        
            for doc_line_dict in test_doc_dict_lst:
                print('输出了 %d 行\n' % ind)
                ind += 1
                f.write(' '.join(str(doc_line_dict[v]) for v in voc) + '\n')
    
            del test_doc_dict_lst
    