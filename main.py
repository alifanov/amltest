import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import json

from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
df = df.set_index('Id')

price = df.SalePrice

sdf = pd.read_csv('test.csv')
sdf = sdf.set_index('Id')

df = df.drop('SalePrice', axis=1)
all_df = df.append(sdf)


all_features = 'MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition'.split(',')
numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','TotalBsmtSF','Fireplaces', 'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
categorical_features = [f for f in all_features if not(f in numeric_features)]

numeric_df = all_df[numeric_features]
X = numeric_df.as_matrix()
imp = pp.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(X)
X = imp.transform(X)

scaler = pp.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

def process_categorical(ndf, df, categorical_features):
    for f in categorical_features:
        new_cols = pd.DataFrame(pd.get_dummies(df[f]))
        new_cols.index = df.index
        ndf = pd.merge(ndf, new_cols, how = 'inner', left_index=True, right_index=True)
    return ndf

numeric_df = pd.DataFrame(X)
numeric_df.index = all_df.index
combined_df = process_categorical(numeric_df, all_df, categorical_features)

X = combined_df.as_matrix()

from sklearn.decomposition import PCA

test_n = df.shape[0]

pca = PCA()
pca.fit(X[:test_n,:], price)
X = pca.transform(X)

X_train = X[:test_n,:]
X_train, X_val, y_train, y_val = ms.train_test_split(X_train, price, test_size=0.3, random_state=0)
X_test = X[test_n:,:]


# housing = load_boston()
# X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
#                                                     train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
y_predicted = tpot.predict(X_test)
sdf['SalePrice'] = y_predicted
sdf.to_csv('submission.csv')
# tpot.export('tpot_kaggle_housing_pipeline.py')