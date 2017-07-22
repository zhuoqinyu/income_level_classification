# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import warnings
warnings.filterwarnings("ignore")
import  numpy as np
import pandas as pd
from scipy import stats
import sklearn as sk
import itertools 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
##  load data; url:https://www.analyticsvidhya.com/blog/2016/09/this-machine-learning-project-on-imbalanced-data-can-add-value-to-your-resume/
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
##  initial exploration
#print(train.head(8))
#print(train.describe())
#print(train.isnull().sum())
#print(test.info())
#

train[["income_level"]]=train[["income_level"]].replace(-50000,0)
train[["income_level"]]=train[["income_level"]].replace(50000,1)
test[["income_level"]]=test[["income_level"]].replace(-50000,0)
test[["income_level"]]=test[["income_level"]].replace(50000,1)
combine=pd.concat([train,test])
high=train[train["income_level"]==1]
low=train[train["income_level"]==0]
high_col="blue"
low_col="red"
print("Higher than 50k: %i (%.1f percent), Lower than 50k: %i (%.1f percent), Total: %i"\
      %(len(high),1.*len(high)/len(train)*100.0,\
        len(low),1.*len(low)/len(train)*100,len(train)))
# separate numerical and categorical variables
num_train =train.select_dtypes(include=['int64'])
num_test=test.select_dtypes(include=['int64'])
cat_train=train.select_dtypes(include=['object'])
cat_test=test.select_dtypes(include=['object'])
#==============================================================================
# distributions of the individual feature
#==============================================================================
#
#plt.figure(figsize=[8,4])
#plt.subplot(121)
#sns.distplot(high['age'].dropna().values, bins=range(0, 81, 1), kde=False, color=high_col,label="> 50k")
#sns.distplot(low['age'].dropna().values, bins=range(0, 81, 1), kde=False, color=low_col,label="< 50k",axlabel='Age')
#plt.legend()
##sns.distplot(train['age'].dropna().values, bins=range(0, 80, 1), kde=True, color='red',axlabel='Age',aylabel='Density')
## do you think population below age 20 could earn >50K under normal circumstances?  Therefore, we can bin this variable into age groups.
#plt.subplot(122)
##industry_code 0 corresponds to zero wage per hour using command:train.loc[train.industry_code==0,["wage_per_hour"]]
#sns.distplot(high['industry_code'].dropna().values, bins=range(1, 52, 1), kde=False, color=high_col,label="> 50k")
#sns.distplot(low['industry_code'].dropna().values, bins=range(1, 52, 1), kde=False, color=low_col,axlabel='Industry_code',label="< 50k")
#plt.legend()
#plt.show()

# The percent of capital_gains!=0 in train is 3.7% ,19.5%, 2.7%
# The percent of capital_losses!=0 in train is 2.0% ,9.4%, 1.5%
#sns.distplot(high['capital_gains'].dropna().values, bins=range(0, 4000, 100), kde=False, color=high_col)
#sns.distplot(low['capital_gains'].dropna().values, bins=range(0,4000, 100), kde=False, color=low_col,axlabel='capital_loss')
#plt.subplot(333)

#plt.figure(figsize=[8,8])
#plt.subplot(211)
#sns.countplot(y="class_of_worker",hue="income_level",data=train)
#plt.subplot(212)
#sns.countplot(y="education",hue="income_level",data=train)
#plt.tight_layout()
#==============================================================================
# Relations between features
#==============================================================================
#plt.figure(figsize=(45,43))
#foo=sns.heatmap(num_train.corr(),vmax=0.6,square=True, annot=True,annot_kws={"size":8})
#plt.xticks(rotation=60)
#plt.yticks(rotation=30)
#plt.tight_layout()
plt.figure(figsize=(45,43))
foo=sns.heatmap(cat_train.corr(),vmax=0.6,square=True, annot=True,annot_kws={"size":8})
plt.xticks(rotation=60)
plt.yticks(rotation=30)
plt.tight_layout()