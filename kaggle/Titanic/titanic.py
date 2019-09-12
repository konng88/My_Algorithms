# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__))

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__))

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)


from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import cross_val_score
#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


data = [train,test]

for df in data:
    df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
mean = df[['Fare','Pclass','Embarked']].groupby(['Pclass','Embarked'])['Fare'].mean()

for df in data:
    mean_age_for_class=df[['Age','Pclass']].groupby(['Pclass'])['Age'].mean()
    df['Age']=df[['Age','Pclass']].apply(lambda x:mean_age_for_class.get(x['Pclass']) if pd.isnull(x['Age']) else x['Age'],axis=1)

for df in data:
    mean_fare_for_class=df[['Fare','Pclass','Embarked']].groupby(['Pclass','Embarked'])['Fare'].mean()
    df['Fare']=df[['Fare','Pclass','Embarked']].apply(lambda x:mean_fare_for_class.get(x['Pclass']).get(x['Embarked']) if pd.isnull(x['Fare']) else x['Fare'],axis=1)


for df in data:
    df['FamilySize'] = df ['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0



def title_picker(title):
    if title not in ['Mr','Mrs','Miss','Master']:
        title='Other'
    return title

for df in data:
    df['Title']=df['Name'].apply(lambda x:x[x.index(',')+1:x.index('.')].strip())
    df['Title']=df['Title'].apply(func=title_picker)


for df in data:
    df['IsMother'] = 0
    df['IsMother'].ix[df[df['Sex']=='female'][df['Title']=='Mrs'][df['Parch']>0].index]=1


for df in data:
    df['IsFather'] = 0
    df['IsFather'].ix[df[df['Sex']=='male'][df['Title']=='Mr'][df['Parch']>0].index]=1

for df in data:
    df['IsSister'] = 0
    df['IsSister'].ix[df[df['Sex']=='female'][df['SibSp']>0].index]=1

for df in data:
    df['IsBrother'] = 0
    df['IsBrother'].ix[df[df['Sex']=='male'][df['SibSp']>0].index]=1



for df in data:
    df['FareBin']=pd.qcut(train['Fare'],4)

for df in data:
    df['AgeBin']=pd.cut(train['Age'],5)



def transform(features,total):
    num_train=len(total[0])
    num_test=len(total[1])
    y_train=total[0]['Survived']
    total[0].drop(['Survived'],axis=1,inplace=True)
    data=pd.concat([total[0],total[1]],axis=0)
    data.index=range(len(data))
    new_data=[]
    new_labels=[]
    num_data=len(data)
    for feature in features:
        uni_feature=data[feature].unique()
        for this_feature in uni_feature:
            label=str(feature)+'_'+str(this_feature)
            new_labels.append(label)
            this_column=np.zeros((num_data,1)).ravel()
            for i in range(0,num_data):
                if data[feature][i]==this_feature:
                    this_column[i]=1
            new_data.append(this_column)
    new_data=pd.DataFrame(np.array(new_data).T,columns=new_labels,dtype='int')
    transformed_data=data.drop(features,axis=1)
    transformed_data=pd.concat([transformed_data,new_data],axis=1)
    X_train=transformed_data.head(num_train)
    X_test=transformed_data.tail(num_test)
    return X_train,X_test,y_train


# data=transform(['Sex','Title','Embarked'],total=data)
#

label=preprocessing.LabelEncoder()
for df in data:
    df['Sex_Code'] = label.fit_transform(df['Sex'])
    df['Embarked_Code'] = label.fit_transform(df['Embarked'])
    df['Title_Code'] = label.fit_transform(df['Title'])
    df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])
    df['FareBin_Code'] = label.fit_transform(df['FareBin'])


features_cat=['Sex','Embarked','Title']

y_train=train['Survived']
X_train=train.drop(['Survived'],axis=1)
X_test=test
num_train=len(X_train)
num_test=len(X_test)
X_total=pd.concat([X_train,X_test],axis=0)
X_dummy=pd.get_dummies(X_total[features_cat])
X_total=pd.concat([X_total,X_dummy],axis=1)

X_total[[]]

features_choices=[]
features_choices.append(['Pclass','Age','SibSp','Parch','Fare','FamilySize','IsAlone','Sex_Code','Embarked_Code','Title_Code','AgeBin_Code','FareBin_Code'])
features_choices.append(['Pclass','Age','SibSp','Parch','Fare','FamilySize','IsAlone','Title_Mr','Title_Mrs','Title_Miss','Title_Master','Title_Other','Embarked_Q','Embarked_S','Embarked_C','Sex_male','Sex_female'])
features_choices.append(['Pclass','Age','SibSp','Parch','Fare','FamilySize','IsAlone','Sex_Code','Embarked_Code','Title_Code','AgeBin_Code','FareBin_Code','Title_Mr','Title_Mrs','Title_Miss','Title_Master','Title_Other','Embarked_Q','Embarked_S','Embarked_C','Sex_male','Sex_female','IsMother','IsFather','IsSister','IsBrother'])
feature_selected=features_choices[2]

X_selected=X_total[feature_selected]
X_train=X_selected.head(num_train)
X_test=X_selected.tail(num_test)



grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]



vote_est = [
    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),

    #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
    ('gpc', gaussian_process.GaussianProcessClassifier()),

    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    ('lr', linear_model.LogisticRegressionCV()),

    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),

    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
    ('knn', neighbors.KNeighborsClassifier()),

    #SVM: http://scikit-learn.org/stable/modules/svm.html
    ('svc', svm.SVC(probability=True)),

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
   ('xgb', XGBClassifier())

]

grid_param =[{
            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            'n_estimators': grid_n_estimator, #default=50
            'learning_rate': grid_learn, #default=1
            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
            'random_state': grid_seed
            }]

start_total = time.perf_counter()

for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

    print('clf[1]  ',clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
    print('param  ',param)


    start = time.perf_counter()
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = 10, scoring = 'roc_auc')
    best_search.fit(X_train, y_train)
    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param)


run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*10)
