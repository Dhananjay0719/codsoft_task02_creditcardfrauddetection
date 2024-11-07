import os 
import numpy as np 
import tensorflow as tf 
import sklearn as sk 
import matplotlib.pyplot as plt
import pandas as pd 
try:
    train_df = pd.read_csv('/kaggle/input/fraud-detection/fraudTrain.csv')
    test_df = pd.read_csv('/kaggle/input/fraud-detection/fraudTest.csv')
except:
    train_df = pd.read_csv('C:/Users/acer/Desktop/CODSOFT/Credit Card Fraud detection/fraudTrain.csv')
    test_df = pd.read_csv('C:/Users/acer/Desktop/CODSOFT/Credit Card Fraud detection/fraudTest.csv')
train_df.head()
train_df.isnull().sum()
def data_pre(X):
    del_col=['merchant','first','last','street','zip','unix_time','Unnamed: 0','trans_num','cc_num']
    X.drop(columns=del_col,inplace=True)
   
    
    X['trans_date_trans_time']=pd.to_datetime(X['trans_date_trans_time'])
    X['trans_date']=X['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
    X['trans_date']=pd.to_datetime(X['trans_date'])
    
    
    X['dob']=pd.to_datetime(X['dob'])
    
    #Calculate Age of each trans 
    X["age"] = (X["trans_date"] - X["dob"]).dt.days //365

    
    X['trans_month']=X['trans_date'].dt.month
    X['trans_year']=X['trans_date'].dt.year
    
    X['gender']=X['gender'].apply(lambda x : 1 if x=='M' else 0)
    X['gender']=X['gender'].astype(int)
    X['lat_dis']=abs(X['lat']-X['merch_lat'])
    X['long_dis']=abs(X['long']-X['merch_long'])
    X=pd.get_dummies(X,columns=['category'])
    X=X.drop(columns=['city','trans_date_trans_time','state','job','merch_lat','merch_long','lat','long','dob','trans_date'])
    return X
    
train_df_pre=data_pre(train_df.copy())
train_df_pre.head()
x_train=train_df_pre.drop('is_fraud',axis=1)
y_train=train_df_pre['is_fraud']
test_df_pre=data_pre(test_df.copy())
test_df_pre.head()
x_test=test_df_pre.drop('is_fraud',axis=1)
y_test=test_df_pre['is_fraud']
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Step 1: Fit the StandardScaler on the training data
scaler = StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
logistic_regression=LogisticRegression()
logistic_regression.fit(x_train,y_train)
y_pred_logistic = logistic_regression.predict(x_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
accuracy_logistic
random_forest = RandomForestClassifier(random_state=42,n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred_rf = random_forest.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_rf
DecisionTree=DecisionTreeClassifier()
DecisionTree.fit(x_train,y_train)
y_pred_dt = DecisionTree.predict(x_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_dt
print("\nClassification Report for Logistic Regression:\n", classification_report(y_test, y_pred_logistic))
print("\nClassification Report for Decision Tree:\n", classification_report(y_test, y_pred_dt))
print("\nClassification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))
