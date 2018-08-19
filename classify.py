# -*- coding: utf-8 -*
'''
Created on 2018��7��28��

@author: ls
'''
import pdb
from sklearn.model_selection._split import train_test_split
import multiprocessing
from multiprocessing.pool import Pool
from utils import *
import xgboost as xgb

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'f1', 1 - f1_score1(labels, preds)

def writePredTestNewToFile(preds, path):
    with open(path, 'w', encoding = 'utf-8') as f:
        f.write("id,class\n")
        for ind, ele in enumerate(preds.tolist()):
            f.write(str(ind)+","+str(int(ele)+1)+'\n')

if __name__ == '__main__':
    finedTrain_path = 'finedTrain.csv'
    finedTest_path = 'finedTest.csv'
    
    finedTrainDocTerms, trainY = getFinedDocs(finedTrain_path, lines = -1)
    finedTestDocTerms = getFinedDocs(finedTest_path, isTrain=False)
    finedTrainDocTerms.extend(finedTestDocTerms)
    
    train_testX = getFinedData(finedTrainDocTerms)
    trainX, testX = train_testX[:len(trainY)], train_testX[len(trainY):]
    
    x_train,_x_test,y_train,_y_test = train_test_split(trainX, trainY, test_size=0.4, random_state=0)
    x_valid,x_test,y_valid,y_test = train_test_split(_x_test, _y_test, test_size=0.25, random_state=0)
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)  # label可以不要，此处需要是为了测试效果
    param = {'max_depth':5, 'eta':0.5, 'silent':1, 'objective':'multi:softmax', 'num_class':19}  # 参数
    evallist  = [(dtrain,'train'), (dvalid,'valid')]  # 这步可以不要，用于测试效果
    num_round = 2000  # 循环次数
    bst = xgb.train(param, dtrain, num_round, evallist, feval=evalerror)
    dtest = xgb.DMatrix(x_test)
    preds = bst.predict(dtest)
    y_test1 = [int(ele) for ele in y_test]
    preds1 = [int(ele) for ele in preds]
    print("f1_score in test: \n", f1_score1(y_test1, preds1))
    
    dnew = xgb.DMatrix(testX)
    dnew_preds = bst.predict(dnew)
    writePredTestNewToFile(dnew_preds, path = 'test_commit.csv')
    
    
    
    
    
    
    

    