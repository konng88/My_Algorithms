import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier



def process(df):
    #fill the nan in train['Embarked'] with mode of Embarked
    mode_embarked=df['Embarked'].mode()[0]
    df['Embarked'].fillna(value=mode_embarked,inplace=True)

    #fill the nan in train['Age'] with the mean in each class
    class_median_age_series = df.groupby(['Pclass'])['Age'].median()
    df1=df[['Age', 'Pclass']].apply(lambda x: class_median_age_series.get(x['Pclass']) if(pd.isnull(x['Age'])) else x['Age'], axis=1)
    df['Age']=df1

    #group the age to 6 groups
    # def age_grouper(age):
    #     if age < 10:
    #         return 1
    #     elif (age >= 10 and age < 20):
    #         return 2
    #     elif (age >= 20 and age < 35):
    #         return 3
    #     elif (age >= 35 and age < 50):
    #         return 4
    #     elif (age >= 50 and age < 65):
    #         return 5
    #     else:
    #         return 6
    # df['Age']=df['Age'].apply(func=age_grouper)

    #transform Title
    df['Title']=df['Name'].apply(lambda x:x[x.index(',')+1:x.index('.')].strip())
    df.drop(['Name'],axis=1,inplace=True)
    def title_grouper(title):
        if(title in ['Mrs', 'Miss', 'Master', 'Mr', 'Dr']):
            title = title
        elif(title in ['Ms','Mme']):
            title = 'Mrs'
        elif(title in ['Mlle', 'Lady']):
            title = 'Miss'
        elif(title in ['Don']):
            title = 'Mr'
        else:
            title = 'Other'
        return title

    df['Title']=df['Title'].apply(func=title_grouper)
    #transform Cabin
    df['Cabin']=df['Cabin'].apply(lambda x:str(x)[0] if pd.notnull(x) else 'O')

    #drop Ticket
    df.drop(['Ticket'],axis=1,inplace=True)
    #drop PassengerId
    df.drop(['PassengerId'],axis=1,inplace=True)
    #transform Sex

    #transform Embarked

    #transform Fare
    class_mean_fare_series = df.groupby(['Pclass'])['Fare'].mean()
    df['Fare']=df[['Fare','Pclass']].apply(lambda x:class_mean_fare_series.get(x['Pclass']) if pd.isnull(x['Fare']) else x['Fare'],axis=1)

def feature_expansion(train,test):
    #preprocessing
    num_train=len(train)
    num_test=len(test)
    Y_train=train['Survived']
    X_train=train.drop(['Survived'],axis=1)
    X_test=test
    X_total=pd.concat([X_train,X_test],axis=0)
    #feature_expansion
    object_features=[]
    for feature in X_total.columns:
        if X_total[feature].dtype=='object':
            object_features.append(feature)
            X_total[feature]=pd.Categorical(X_total[feature]).codes

    for feature in object_features:

        col_data=X_total[feature]

        count=len(col_data.unique())
        num_data=len(col_data)
        new_data=np.zeros((num_data,count))

        for i in range(0,num_data):

            new_data[i][col_data.ravel()[i]]=1
        new_data_frame=pd.DataFrame(new_data)

        for col_title in new_data_frame.columns:
            new_title=feature+" "+str(col_title)
            X_total.insert(0,new_title,new_data_frame[col_title])
        X_total.drop([feature],axis=1,inplace=True)
    return Y_train,X_total.head(num_train),X_total.tail(num_test)

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
process(train)
process(test)
data=feature_expansion(train,test)
y_train=data[0]
X_train=preprocessing.scale(data[1])
X_test=preprocessing.scale(data[2])

svm=svm.SVC(gamma='scale',C=80)
svm.fit(X_train,y_train)
y_svm_score = cross_val_score(svm, X_train, y_train, cv=10, scoring='accuracy')
print("SVM accuracy: ",y_svm_score.mean())
rdf=RandomForestClassifier(n_estimators=100)
rdf.fit(X_train,y_train)
y_rdf_score = cross_val_score(rdf, X_train, y_train, cv=10, scoring='accuracy')
print("RandomForestClassifier accuracy: ",y_rdf_score.mean())

y_predict=svm.predict(X_test)

submission=pd.DataFrame()

id=pd.read_csv('test.csv')['PassengerId'].ravel()
submission['PassengerId']=id
submission['Survived']=y_predict

submission.to_csv("submission_new.csv",index=False)
