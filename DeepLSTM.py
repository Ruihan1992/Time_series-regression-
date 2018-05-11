# Time_series-regression-
Some  methods for value of futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.utils import shuffle
import random
df_train=pd.read_excel('G:/招商/X序列_train.xls')
######################################缺失分析###############
df_train=df_train.iloc[1:,:]
na_count = df_train.iloc[:,1:].isnull().sum().sort_values(ascending=False)
na_rate = na_count / len(df_train)
na_data = pd.concat([na_count,na_rate],axis=1,keys=['count','ratio'])
df_train = df_train.drop(na_data[na_data['ratio']>0.60].index, axis=1)  # 删除缺省率大于0.8
#df_train = df_train.drop('中债国债到期收益率:1年',axis=1)
#df_train = df_train.drop('中债国债到期收益率:3个月',axis=1)
#df_train = df_train.drop('中债国债到期收益率:5年',axis=1)

#####################################剩下的补0##################
na_count = df_train.iloc[:,1:].isnull().sum().sort_values(ascending=False)
missing_cols = na_count[na_count>0].index
df_train[missing_cols] = df_train[missing_cols].fillna(0.)  # 从这些变量的意义来看，缺失值很
df_train[missing_cols].isnull().sum()  # 验证缺失值是否都已补全

##############################输入Y并进行anova分析################################
g=0
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
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib import rnn
Supervised=df_train.iloc[Labeled,:]
Unspervised=df_train.iloc[Unlabeled,:]
scaler_for_x=MinMaxScaler(feature_range=(0,1))
scaler_for_y=MinMaxScaler(feature_range=(0,1))
Supervised.iloc[:,1:] = scaler_for_x.fit_transform(Supervised.iloc[:,1:])
Unspervised.iloc[:,1:] = scaler_for_y.fit_transform(Unspervised.iloc[:,1:])
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
Supervised=Supervised.iloc[:,features]
Unspervised=Unspervised.iloc[:,features]
###########################################################
def find_batch(X,test_X,batch_size,time_step,state):
    
    if state==1:
        batch_index=[]
        dx=X[:,:-1]
        dy=X[:,-1]    
        batch_index=[]
        train_x,train_y=[],[]
    
        for i in range(len(X)-time_step):
            if i%batch_size==0:
                batch_index.append(i)
            x=dx[i:i+time_step,:]
            y=dy[i:i+time_step]
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(X)-time_step))
        
        size=(len(test_X)+time_step-1)//time_step
        test_x=[]
        for i in range(size-1):
            x=test_X[i*time_step:(i+1)*time_step,:]
            test_x.append(x.tolist())
        test_x.append((test_X[(i+1)*time_step:,:]).tolist())
        return batch_index,train_x,train_y,test_x
    else:
        batch_index=[]
        dx=X
        train_x=[]
        for i in range(len(X)-time_step):
            if i%batch_size==0:
                batch_index.append(i)            
            x=dx[i:i+time_step,:]
            train_x.append(x.tolist())
        batch_index.append((len(X)-time_step))
        size=(len(test_X)+time_step-1)//time_step
        test_x=[]
        for i in range(size-1):
            x=test_X[i*time_step:(i+1)*time_step,:]
            test_x.append(x.tolist())
        test_x.append((test_X[(i+1)*time_step:,:]).tolist())        
        return batch_index,train_x
            

batch_size=50
time_step=10


###############Construct test_dataset############# 
X_test_book = pd.read_excel('G:/招商/X序列_test.xls')
X_test = X_test_book[Supervised.columns]
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
Supervised_test=scaler_for_x.fit_transform(X_test.iloc[Ind1,:].values)
batch_index,train_x,train_y,test_x=find_batch(np.concatenate((Supervised.values,Y_train.iloc[:,1].values.reshape([len(Y_train),1])),axis=1),Supervised_test,batch_size,time_step,1)
###############Construct LSTM#####################
learning_rate=0.001
training_iters=3000
batch_size=50
display_step=100
n_input=len(features)
n_hidden=10
n_outputs=1
n_layers=4

x=tf.placeholder(tf.float32,[None,time_step,n_input])
y=tf.placeholder(tf.float32,[None,time_step,n_outputs])
weights={'in':tf.Variable(tf.random_normal([n_input,n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden,n_outputs]))}
biases={'in':tf.Variable(tf.constant(0.1,shape=[n_hidden,])),
        'out': tf.Variable(tf.constant(0.1,shape=[n_outputs,]))}

X=train_x
Y=train_y
test=test_x

w_in=weights['in']
b_in=biases['in']
inputs=tf.reshape(x,[-1,n_input])
input_rnn=tf.matmul(inputs,w_in)+b_in
input_rnn=tf.reshape(input_rnn,[-1,time_step,n_hidden])    
lstm_cells=[rnn.LSTMCell(n_hidden,forget_bias=1.0) for _ in range(n_layers)]
lstm=rnn.MultiRNNCell(lstm_cells)
outputs,states=tf.nn.dynamic_rnn(lstm,inputs=x,dtype=tf.float32,time_major=False)
outputs=tf.reshape(outputs,[-1,n_hidden])
w_out=weights['out']
b_out=biases['out']
pred=tf.matmul(outputs,w_out)+b_out
    #损失函数
loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(y, [-1])))
train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
        #重复训练10000次
    for i in range(1000):
        step=0
        start=0
        end=start+batch_size
        while(end<len(X)):
            batch_x=np.array(X)[start:end,:,:].reshape([batch_size,time_step,n_input])
            batch_y=np.array(Y)[start:end,:].reshape([batch_size,time_step,n_outputs])
            _,_loss=sess.run([train_op,loss],feed_dict={x:batch_x,y:batch_y})
            start+=batch_size
            end=start+batch_size
                #每10步保存一次参数
            if step%10==0:
                print(i,step,_loss)
            step+=1        
    predict=[]
    for i in range(len(test)):
        if len(test[i])==time_step:
            batch=np.array(test[i]).reshape([1,time_step,n_input])
            prob=sess.run(pred,feed_dict={x:batch})
            temp=prob.reshape((-1))
            predict.extend(temp)
        else:
            batch=np.concatenate((np.array(test[g]),np.zeros([time_step-len(test[g]),n_input])),axis=0).reshape([1,time_step,n_input])
            prob=sess.run(pred,feed_dict={x:batch})
            temp=prob.reshape((-1))
            predict.extend(temp[:len(test[g])])
    
