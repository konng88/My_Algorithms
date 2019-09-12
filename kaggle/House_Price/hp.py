import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as  sns

from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')




train=pd.read_csv('train.csv')
train_size=len(train)
test=pd.read_csv('test.csv')
test_size=len(test)
Y_train=train['SalePrice']
train.drop(['SalePrice'],axis=1,inplace=True)
data=pd.concat((train,test),axis=0).reset_index(drop=True)
# print(train_size,test_size)

str_vars=['MSSubClass','YrSold','MoSold']
for var in str_vars:
    train[var]=train[var].apply(str)

# print(train['MSSubClass'][0])

common_vars=['Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual']
for var in common_vars:
    data[var].fillna(data[var].mode()[0],inplace=True)

data['MSZoning']=data.groupby('MSSubClass')['MSZoning'].transform(lambda x:x.fillna(x.mode()[0]))

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',
            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC"
           ,'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']:
    data[col] = data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'
           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF'):
           data[col].fillna(data[col].median(),inplace=True)

data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

data['Functional'] = data['Functional'].fillna('Typ')

# print(data.isnull().sum().sort_values(ascending=True))

categorical_features = data.select_dtypes(include=['object']).columns
numerical_features = data.select_dtypes(exclude=['object']).columns
