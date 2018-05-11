# Time_series-regression-
Some  methods for value of futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
df_train=pd.read_excel('G:/招商/X序列_train.xls')
######################################缺失分析###############
df_train=df_train.iloc[1:,:]
na_count = df_train.iloc[:,1:].isnull().sum().sort_values(ascending=False)
na_rate = na_count / len(df_train)
na_data = pd.concat([na_count,na_rate],axis=1,keys=['count','ratio'])
df_train = df_train.drop(na_data[na_data['ratio']>0.80].index, axis=1)  # 删除缺省率大于0.8
#df_train = df_train.drop('中债国债到期收益率:1年',axis=1)
#df_train = df_train.drop('中债国债到期收益率:3个月',axis=1)
#df_train = df_train.drop('中债国债到期收益率:5年',axis=1)

#####################################剩下的补0##################
na_count = df_train.iloc[:,1:].isnull().sum().sort_values(ascending=False)
missing_cols = na_count[na_count>0].index
df_train[missing_cols] = df_train[missing_cols].fillna(0.)  # 从这些变量的意义来看，缺失值很
df_train[missing_cols].isnull().sum()  # 验证缺失值是否都已补全

##############################输入Y并进行anova分析################################
g=2
Y_train = pd.read_excel('G:/招商/Y序列_train.xls',sheetname=g)
#####################################换成时间戳########################################
import xlrd
X_train_book = xlrd.open_workbook('G:/招商/X序列_train.xls')
Table_X_train=X_train_book.sheets()[0]
X_train_nrows = Table_X_train.nrows  #行数
X_time=[]
for i in range(2,X_train_nrows):  ####头两个是单位,删
    rowValues= Table_X_train.row_values(i) #某一行数据
    X_time.append(rowValues[0])
Y1_train_book = xlrd.open_workbook('G:/招商/Y序列_train.xls')
Table_Y1_train=Y1_train_book.sheets()[g]  #Y1,Y2,Y3
Y_train_nrows = Table_Y1_train.nrows  #行数
Y_time=[]
for i in range(1,Y_train_nrows):
    rowValues = Table_Y1_train.row_values(i) #某一行数据
    Y_time.append(rowValues[0])     
Labeled=[]
Unlabeled=[]    
for i in range(len(X_time)):
    if X_time[i] in Y_time:
        Labeled.append(i)
    else:
        Unlabeled.append(i)
  
############################ 非线性模型特征排序#########################################  
Supervised=df_train.iloc[Labeled,:]
Unspervised=df_train.iloc[Unlabeled,:]
Supervised.iloc[:,1:] = StandardScaler().fit_transform(Supervised.iloc[:,1:])
Unspervised.iloc[:,1:] = StandardScaler().fit_transform(Unspervised.iloc[:,1:])
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=50, max_depth=4)
scores = []
features = []
for i in range(1,Supervised.shape[1]):
    score = cross_val_score(rf, Supervised.iloc[:, i:i+1], Y_train.iloc[:,1], scoring="r2",cv=ShuffleSplit(len(Supervised), 3, .3))
    scores.append((format(np.mean(score), '.3f'), i))
    if abs(np.mean(score))>0.1:
        features.append(i)

############################ 伪标签#########################################  
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
import random
class PseudoLabeler(BaseEstimator, RegressorMixin):
    
    def __init__(self, model, test, features, target, sample_rate=0.2, seed=42):
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed
        
        self.test = test
        self.features = features
        self.target = target
        
    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "test": self.test,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        
    def fit(self, X, y):
        if self.sample_rate > 0.0:
            augemented_train = self.__create_augmented_train(X, y)
            self.model.fit(
                augemented_train[:,self.features],
                augemented_train[:,-1]
            )
        else:
            self.model.fit(X[:,self.features], y)
        
        return self


    def __create_augmented_train(self, X, y):
        num_of_samples = int(len(self.test) * self.sample_rate)
        
        # Train the model and creat the pseudo-labels
        self.model.fit(X[:,self.features], y)
        pseudo_labels = self.model.predict(self.test[:,self.features])
        
        # Add the pseudo-labels to the test set
        augmented_test = np.concatenate((self.target, pseudo_labels),axis=0)
        augmented_test = np.column_stack((np.concatenate((X,self.test),axis=0),augmented_test))
        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        #sampled_test = augmented_test.sample(n=num_of_samples)
        sampled_test = np.array(random.sample(augmented_test.tolist(), num_of_samples))
        temp_train = np.column_stack((X, y))
        augemented_train = np.row_stack([sampled_test, temp_train])

        return shuffle(augemented_train)
        
    def predict(self, X):
        return self.model.predict(X[:,self.features])
    
    def get_model_name(self):
        return self.model.__class__.__name__ 
    
from sklearn.cross_validation import cross_val_score
model_factory=[RandomForestRegressor(),XGBRegressor(nthread=2,eval_metric="rmse",eta=0.06,max_depth=6,min_child_weight=5,gamma=0.2),ExtraTreesRegressor(),GradientBoostingRegressor()]
RMSE=np.zeros(len(model_factory))
PL=[[],[],[],[]]
for i in range(len(model_factory)):
    model_factory[i].seed = 42
    num_folds = 8
    PL[i] = PseudoLabeler(model_factory[i],Unspervised.values,features,Y_train.iloc[:,1].values,sample_rate=0.2, seed=50)
    PL[i].fit(Supervised.values,Y_train.iloc[:,1].values)    
    scores = cross_val_score(model_factory[i],Supervised.iloc[:,features].values, Y_train.iloc[:,1].values, cv=num_folds, scoring='neg_mean_squared_error')
    score_description = " %0.2f (+/- %0.2f)" % (np.sqrt(scores.mean()*-1), scores.std() * 2)
    print('{model:25} CV-5 RMSE: {score}'.format(model=model_factory[i].__class__.__name__,score=score_description ))

############################################ Step 3: conbin###############################
Ind=RMSE.tolist().index(min(RMSE)) #####最好分类器##### 
X_test_book = pd.read_excel('G:/招商/X序列_test.xls')
X_test = X_test_book[Supervised.columns]
#######################################################################################
X_test_book = xlrd.open_workbook('G:/招商/X序列_test.xls')
Table_X_test=X_test_book.sheets()[0]
X_test_nrows = Table_X_test.nrows  #行数
X_test_time=[]
for i in range(2,X_test_nrows):  ####头两个是单位,删
    rowValues= Table_X_test.row_values(i) #某一行数据
    X_test_time.append(rowValues[0])
    
Y1_test_book = xlrd.open_workbook('G:/招商/Y序列_test.xlsx')
Table_Y1_test=Y1_test_book.sheets()[g+1]
Ind1=[]
for i in range(Table_Y1_test.nrows):
    rowValues = Table_Y1_test.row_values(i) #某一行数据
    Ind1.append(np.array(X_test_time).tolist().index(rowValues[0]))

X_test = X_test.iloc[1:,:].fillna(0.)  # 从这些变量的意义来看，缺失值很    
X_test.iloc[:,1:] = StandardScaler().fit_transform(X_test.iloc[:,1:])    
Predict=PL[Ind].predict(X_test.iloc[Ind1,:].values)
