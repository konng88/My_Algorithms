import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm

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

def test(gamma,C):
    data_train_total=load('train.csv')
    package_train=process_format(data_train_total,train_size=500)
    data_train=package_train[0]
    data_test=package_train[1]
    labels_train=package_train[2]
    labels_test=package_train[3]
    clf=svm.SVC(gamma=gamma,C=C)
    clf.fit(data_train,labels_train)
    train_score=clf.score(data_train,labels_train)
    test_score=clf.score(data_test,labels_test)

    return train_score,test_score

gamma_list=[]
for gamma in range(1,10):
    gamma_list.append(gamma)
C_list=[]
for C in range(1,10):
    C_list.append(C)
for gamma in gamma_list:
    for C in C_list:
        package=test(gamma,C)
        if abs(package[0] - package[1])<0.03:
                print("-------------result for rbf: gamma = "+str(gamma)+" C = "+str(C) +" is-------------")
                print("accuracy on training set = " +str(package[0]))
                print("accuracy on testing set = " +str(package[1]))
