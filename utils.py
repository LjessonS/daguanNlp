# -*-coding:utf-8-*-
'''
Created on 2018骞�8鏈�4鏃� 涓嬪崍7:59:37

@author: liusheng1
'''

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