import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn import preprocessing
import pandas as pd
import numpy as np
import math
import xgboost as xgb
np.random.seed(2019)
from scipy.stats import skew
from scipy import stats
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import copy
import statsmodels
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


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
X_train=preprocessing.scale(X_train)


#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(learning_rate=0.1,n_estimators=300,random_state=0),
    ensemble.BaggingClassifier(max_samples= 0.25, n_estimators= 300, random_state= 0),
    ensemble.ExtraTreesClassifier(criterion= 'entropy', max_depth= 6, n_estimators= 100, random_state= 0),
    ensemble.GradientBoostingClassifier(learning_rate= 0.05, max_depth= 2, n_estimators= 300, random_state= 0),
    ensemble.RandomForestClassifier(criterion= 'entropy', max_depth= 6, n_estimators= 100, oob_score= True, random_state= 0),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(max_iter_predict= 10, random_state= 0),

    #GLM
    linear_model.LogisticRegressionCV(fit_intercept= True, random_state= 0, solver= 'liblinear'),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    #Navies Bayes
    naive_bayes.BernoulliNB(alpha= 0.1),
    naive_bayes.GaussianNB(),

    #Nearest Neighbor
    neighbors.KNeighborsClassifier(algorithm= 'brute', n_neighbors= 7, weights= 'uniform'),

    #SVM
    svm.SVC(C= 2, decision_function_shape= 'ovo', gamma= 0.1, probability= True, random_state= 0),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    #Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),


    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier(learning_rate= 0.01, max_depth= 4, n_estimators= 300, seed= 0)
    ]

for alg in MLA:
    alg.fit(X_train,y_train)
    score = cross_val_score(alg, X_train, y_train, cv=10, scoring='accuracy')
    print(alg.__class__.__name__,": ",score.mean())


# y_test=xgb.predict(X_test)
# submit=X_total[['PassengerId']].tail(num_test)
# submit['Survived']=y_test
# submit.to_csv("submission_tutorial.csv",index=False)
