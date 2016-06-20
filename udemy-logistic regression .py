
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from pandas import DataFrame, Series #dataframe 


# In[31]:

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')
 
#plot


# In[3]:

import math #math 


# In[5]:

from sklearn.linear_model import LogisticRegression 
from sklearn.cross_validation import train_test_split 
#machine learning 


# In[6]:

from sklearn import metrics #evaluate results 


# In[8]:

import statsmodels.api as sm #dataset 


# In[42]:

df = sm.datasets.fair.load_pandas().data


# Number of observations: 6366
# Number of variables: 9
# Variable name definitions:
# 
#     rate_marriage   : How rate marriage, 1 = very poor, 2 = poor, 3 = fair,
#                     4 = good, 5 = very good
#     age             : Age
#     yrs_married     : No. years married. Interval approximations. See
#                     original paper for detailed explanation.
#     children        : No. children
#     religious       : How relgious, 1 = not, 2 = mildly, 3 = fairly,
#                     4 = strongly
#     educ            : Level of education, 9 = grade school, 12 = high
#                     school, 14 = some college, 16 = college graduate,
#                     17 = some graduate school, 20 = advanced degree
#     occupation      : 1 = student, 2 = farming, agriculture; semi-skilled,
#                     or unskilled worker; 3 = white-colloar; 4 = teacher
#                     counselor social worker, nurse; artist, writers;
#                     technician, skilled worker, 5 = managerial,
#                     administrative, business, 6 = professional with
#                     advanced degree
#     occupation_husb : Husband's occupation. Same as occupation.
#     affairs         : measure of time spent in extramarital affairs
# 
# See the original paper for more details.

# In[215]:

df.head()


# build a new column

# In[61]:

def affair_check(x):
    if x != 0:
        return 1 
    else:
        return 0
df['Had_affair'] = df['affairs'].apply(affair_check)


# In[62]:

df.head()


# In[40]:

df.groupby('Had_affair').mean()


# In[66]:

sns.factorplot('age',data=df,hue='Had_affair',palette='coolwarm')


# In[65]:

sns.factorplot('yrs_married',data=df,hue='Had_affair',palette='coolwarm')


# In[217]:

sns.factorplot('children',data=df,hue='Had_affair',palette='coolwarm')


# In[52]:

occ_dummies = pd.get_dummies(df['occupation'])


# In[54]:

hus_occ_dummies = pd.get_dummies(df['occupation_husb'])


# In[57]:

occ_dummies.head()


# In[56]:

occ_dummies.columns = ['occ1','occ2','occ3','occ4','occ5','occ6']


# In[58]:

hus_occ_dummies.columns = ['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']


# In[176]:

X = df.drop(['occupation','occupation_husb','Had_affair'],axis=1)


# In[177]:

dummies = pd.concat([occ_dummies,hus_occ_dummies],axis=1)


# In[178]:

X = pd.concat([X,dummies],axis = 1)


# In[179]:

X.head()


# In[180]:

Y = df.Had_affair


# In[181]:

Y.tail()


# In[182]:

X = X.drop('affairs',axis=1)


# In[185]:

X = X.drop('coefficient',axis = 1)


# In[193]:

X.head()
X = X.drop('occ1',axis = 1)


# In[194]:

X = X.drop('hocc1',axis = 1)


# In[195]:

X.head()


# In[196]:

Y = np.ravel(Y)

Y


# In[197]:

log_model = LogisticRegression()

log_model.fit(X,Y)

log_model.score(X,Y)


# In[198]:

Y.mean()


# In[199]:

coeff = log_model.coef_
coeff


# In[200]:

def transform (x):
    for i in x:
        return i 
        


# In[201]:

variables = pd.DataFrame(data = list(X.columns.values))
coeff = pd.DataFrame(data = coeff)


# In[202]:

variables['coefficient'] = coeff.apply(transform)


# In[203]:

variables


# In[204]:

X_train,X_test,Y_train,Y_test = train_test_split(X,Y)


# In[205]:

log_model2 = LogisticRegression()

log_model2.fit(X_train,Y_train)


# In[206]:

class_predict = log_model2.predict(X_test)


# In[209]:

print (metrics.accuracy_score(Y_test,class_predict))


# In[239]:

p = np.array([1,45,20,1,2,15,1,0,0,0,0,1,0,0,0,0])
predict = log_model.predict(p)


# In[240]:

predict


# In[ ]:



