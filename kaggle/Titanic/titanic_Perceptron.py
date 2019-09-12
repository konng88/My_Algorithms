import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import Perceptron as p

def load(file_name):
    col_labels=[]
    with open(file_name) as file:
        col_labels=file.readline().split(',')
    data_total=pd.read_csv(file_name)
    data_total.columns = col_labels
    data_total.drop(['PassengerId'],axis=1,inplace=True)
    data_total.drop(['Cabin'],axis=1,inplace=True)
    data_total.drop(['Name'],axis=1,inplace=True)
    data_total.drop(['Ticket'],axis=1,inplace=True)
    data_total=data_total.dropna()
    return data_total

def process_format(data_total,train_size):
    labels_train=data_total['Survived'].head(train_size).ravel()
    labels_test=data_total['Survived'].tail(len(data_total)-train_size).ravel()
    data_total.drop(['Survived'],axis=1,inplace=True)
    for feature in data_total.columns:
        if data_total[feature].dtype=='object':
            data_total[feature]=pd.Categorical(data_total[feature]).codes

    num_features=data_total.shape[1]-1
    data_train=data_total.iloc[:,0:num_features].head(train_size)
    data_test=data_total.iloc[:,0:num_features].tail(len(data_total)-train_size)
    data_train=preprocessing.scale(np.array(data_train))
    data_test=preprocessing.scale(np.array(data_test))
    return data_train,data_test,labels_train,labels_test

def test():
    data_train_total=load('train.csv')
    package_train=process_format(data_train_total,train_size=500)
    data_train=package_train[0]
    data_test=package_train[1]
    labels_train=package_train[2]
    labels_test=package_train[3]
    clf=p()
    clf.fit(data_train,labels_train)
    train_score=clf.score(data_train,labels_train)
    test_score=clf.score(data_test,labels_test)
    print("-------------result for perceptron is-------------")
    print("accuracy on training set = " +str(train_score))
    print("accuracy on testing set = " +str(test_score))

test()
