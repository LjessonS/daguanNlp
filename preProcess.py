# -*- coding: utf-8 -*
'''
Created on 2018��7��28��

@author: ls
'''
import pdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import ldamulticore
from numpy import log10
from multiprocessing.pool import Pool
import multiprocessing

# train_path = 'd:/VM_Share/work/codeSpace/nlp/daguan/new_data/train_set.csv'
# test_path = 'd:/VM_Share/work/codeSpace/nlp/daguan/new_data/test_set.csv'
train_path = 'train_set_sample.csv'
test_path = 'test_set_sample.csv'
train_out_path = 'trainT.csv'
test_out_path = 'testT.csv'

def extract_docs(texts, isTrain = True):
    header = []
    docs = []
    y = []
    with open(texts) as f:
        for ind, line in enumerate(f):
            if ind == 0:
                header = line.strip().split(',')
                continue
            line_tmp = line.strip().split(',')
            docs.append(line_tmp[2])
            if isTrain:
                y.append(line_tmp[-1])
    if isTrain:            
        return docs, y
    else:
        return docs

train_docs, train_y = extract_docs(train_path)
test_docs = extract_docs(train_path, False)

train_len = len(train_docs)
cntVector = CountVectorizer()
global cntTf
cntTf = cntVector.fit_transform(train_docs + test_docs)

def outPutFeature(data, filePath, Y = None, isTrain = True):
    with open(filePath, 'w') as f:
        if isTrain:
            for feature, y in zip(data, Y):
                f.write(' '.join(str(ele) for ele in feature) + ',' + y + '\n')
        else:
            for feature in data:
                f.write(' '.join(str(ele) for ele in feature) + '\n')

def phred_scale(ele):
    return -10*log10(ele)  

def operate(k):
    print("now running topic: %s" % k)
    lda = LatentDirichletAllocation(n_topics=k,
                                    max_iter = 200,
                                    learning_offset=50.,
                                    random_state=0, evaluate_every=1)
    docres = lda.fit_transform(cntTf)
    p = lda.perplexity(cntTf)
    
    return p, docres

if __name__ == '__main__':
    import time
    t0 = time.time()
    
    topicNumLst = range(5, 300, 3)
    perplexity = []
    min_perplexity = 999999

    cpus = multiprocessing.cpu_count()
    pool = Pool(cpus-1)
    perplexity_docres_lst = pool.map(operate, topicNumLst)
    pool.close()
    pool.join()

    perplexityLst = [e[0] for e in perplexity_docres_lst]
    bestModelInd = perplexityLst.index(min(perplexityLst))
    features = perplexity_docres_lst[bestModelInd][1]
    del perplexity_docres_lst
    outPutFeature(features[:train_len], train_out_path, train_y)
    outPutFeature(features[train_len:], test_out_path, isTrain=False)
#     pdb.set_trace()

    t1 = time.time()
    print("花费了   %f 小时\n" % ((t1 - t0)/3600))


