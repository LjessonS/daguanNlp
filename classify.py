# -*- coding: utf-8 -*
'''
Created on 2018��7��28��

@author: ls
'''
import pdb
from sklearn.model_selection._split import train_test_split
import multiprocessing
from multiprocessing.pool import Pool
from sklearn import metrics

def read_dat(path, isTrain = True):
    X = []
    Y = []
    with open(path) as f:
        for line in f:
            line_tmp = line.strip().split(',')
            X.append(float(ele) for ele in line_tmp[0].split(' '))
            if isTrain:
                Y.append(int(line_tmp[-1]))
    
    if isTrain:
        return X, Y
    return X

def mean_inner_product(X, y):
    s = 0
    length = len(X)
    for x in X:
        s += sum([i*j for i,j in zip(x,y)])
    return s / length

def getClassDict(trainY):
    classDict = {}
    
    for ind, i in enumerate(trainY):
        if i not in classDict:
            classDict[i] = [ind]
        else:
            classDict[i].append(ind)
            
    return classDict

def getTrainGroup(trainX, classDict):
    sortedKeys = sorted([key for key in classDict])
    trainXX = []
    for key in sortedKeys:
        group = []
        for ind in classDict[key]:
            group.append(trainX[ind])
        trainXX.append(group)
    return trainXX

def predictY(X, trainX, clsDict):
    keyLst = sorted([key for key in clsDict])
    trainXX = getTrainGroup(trainX, clsDict)
    
    pred_y = []
    for x in X:
        distLst = [mean_inner_product(trainX, x) for trainX in trainXX]
        
        label_ind = distLst.index(min(distLst))
        pred_y.append(keyLst[label_ind])
    return pred_y



if __name__ == '__main__':
    train_out_path = 'trainT.csv'
    pred_out_path = 'testT.csv'
    predY_out_path = 'testY.csv'
    
    trainX, trainY = read_dat(train_out_path)
    x_train,x_test,y_train,y_test = train_test_split(trainX, trainY, test_size=0.3, random_state=0)
    predX = read_dat(pred_out_path, False)
    
    
    clsDict = getClassDict(trainY)
    pred_trainY = predictY(x_test, trainX, clsDict)
    
    print("F1 score: %.3f" % metrics.f1_score(y_test, pred_trainY, average='weighted'))
    
    pred_Y = predictY(predX, trainX, clsDict)
    with open(predY_out_path, 'w') as f:
        f.write('\n'.join(str(ele) for ele in pred_Y))
    
    
    
    

    