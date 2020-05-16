#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[94]:


import zipfile
with zipfile.ZipFile('train_yhhx1Xs.zip','r') as zipref:
    zipref.extractall()


# In[125]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test_QkPvNLx.csv')
sample = pd.read_csv('sample_submission_pn2DrMq.csv')


# # Exploratory Data Analysis

# In[14]:


#analysis of training set
print(train.shape)
print(test.shape)


# In[18]:


traintest = pd.concat([train,test],axis = 0)


# In[19]:


traintest.head()


# In[21]:


traintest.nunique()


# In[23]:


for col in traintest.columns[3:8]:
    print(col, train[col].unique())


# # Univariate Analysis

# In[8]:


train.columns


# In[24]:


cat = ['Course_Domain', 'Course_Type',
       'Short_Promotion', 'Public_Holiday', 'Long_Promotion']


# In[25]:


cont = ['ID', 'Day_No', 'Course_ID',  'User_Traffic',
       'Competition_Metric', 'Sales']


# In[26]:


train.groupby(by = 'Course_Type')['Course_Type'].count()


# # Making a baseline model

# In[12]:


#Imputation
train['Competition_Metric'].fillna(train['Competition_Metric'].mean(), inplace = True)


# In[ ]:





# In[ ]:





# In[199]:


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:-1],train.iloc[:,-1],test_size =0.3, random_state = 42 )


# In[ ]:





# In[198]:


# Baseline
y_train.mean()


# In[149]:


y_pred = np.ones(y_test.shape)*y_train.mean()


# In[153]:


y_pred = pd.Series(y_pred)


# In[156]:


from sklearn import metrics


# In[157]:


metrics.mean_squared_error(y_test, y_pred)


# In[158]:


metrics.r2_score(y_test, y_pred)


# In[177]:


pd.DataFrame(y_pred).join(y_test.reset_index(drop =True))


# In[179]:


y_test = y_test.reset_index(drop =True)


# In[182]:


metrics.r2_score(y_test, y_pred)


# In[186]:


sns.scatterplot(np.arange(1,100,1),y_test[1:100] )


# In[187]:


sns.scatterplot(np.arange(1,100,1),y_pred[1:100])


# In[202]:


#One hot encode
encoded_col = ['Course_Domain', 'Course_Type']


# In[204]:


train = pd.get_dummies(train,columns = encoded_col)


# In[220]:


train.columns


# In[258]:


col = ['Day_No', 'Course_ID', 'Short_Promotion', 'Public_Holiday',
       'Long_Promotion', 'Competition_Metric',
       'Course_Domain_Business', 'Course_Domain_Development',
       'Course_Domain_Finance & Accounting',
       'Course_Domain_Software Marketing', 'Course_Type_Course',
       'Course_Type_Degree', 'Course_Type_Program']


# In[259]:


X = train[col]


# In[260]:


y = train['Sales']


# In[261]:


X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[104]:


# Linear Regression
from sklearn import metrics
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)
metrics.r2_score(y_test,y_pred_lr)


# In[105]:


# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
metrics.r2_score(y_test,y_pred_rf)


# In[106]:


from xgboost import XGBRegressor


# In[299]:


model_xg = XGBRegressor()
model_xg.fit(X_train,y_train)
y_pred_xg = model_xg.predict(X_test)


# In[319]:


metrics.r2_score(y_test, y_pred_xg)


# In[305]:


sub_encode_col = ['Course_Domain','Course_Type']


# In[306]:


test.head()


# In[307]:


test = pd.get_dummies(test,columns = sub_encode_col)


# In[119]:


test.drop('ID', axis= 1, inplace = True)


# In[120]:


test['Competition_Metric'].fillna(test['Competition_Metric'].mean(),inplace = True)


# In[117]:


test = pd.get_dummies(test,columns = encoded_col)


# In[121]:


sub_y_pred_lr = model_lr.predict(test)
sub_y_pred_rf = model_rf.predict(test)
sub_y_pred_xg = model_xg.predict(test)


# In[122]:


submission_lr = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_lr))
submission_rf = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_rf))
submission_xg = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_xg))


# In[ ]:


submission_lr = submission_lr.rename(columns={'ID':'ID',0:'Sales'})
submission_rf = submission_rf.rename(columns={'ID':'ID',0:'Sales'})
submission_xg = submission_xg.rename(columns={'ID':'ID',0:'Sales'})


# In[299]:


model_xg = XGBRegressor()
model_xg.fit(X_train,y_train)
y_pred_xg = model_xg.predict(X_test)


# In[319]:


metrics.r2_score(y_test, y_pred_xg)


# In[305]:


sub_encode_col = ['Course_Domain','Course_Type']


# In[306]:


test.head()


# In[307]:


test = pd.get_dummies(test,columns = sub_encode_col)


# In[308]:


test.drop('ID', axis= 1, inplace = True)


# In[313]:


test['Competition_Metric'].fillna(test['Competition_Metric'].mean(),inplace = True)


# In[314]:


test.columns


# In[315]:


sub_y_pred_lr = model_lr.predict(test)
sub_y_pred_rf = model_rf.predict(test)
sub_y_pred_xg = model_xg.predict(test)


# In[316]:


submission_lr = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_lr))
submission_rf = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_rf))
submission_xg = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_xg))


# In[317]:


submission_lr = submission_lr.rename(columns={'ID':'ID',0:'Sales'})
submission_rf = submission_rf.rename(columns={'ID':'ID',0:'Sales'})
submission_xg = submission_xg.rename(columns={'ID':'ID',0:'Sales'})


# <img src = 'Capture.JPG'>

# In[320]:


np.sqrt(metrics.mean_squared_log_error(y_test,y_pred_lr))


# In[321]:


np.sqrt(metrics.mean_squared_log_error(y_test,y_pred_rf))


# In[ ]:





# # Improving the model

# In[32]:


train.drop('User_Traffic', axis = 1, inplace = True)


# In[66]:


traintest = pd.concat([train, test], axis =  0)


# In[67]:


traintest.info()


# In[68]:


traintest['Competition_Metric'].fillna(traintest['Competition_Metric'].mean(), inplace = True)


# In[69]:


traintest.info()


# In[39]:


from sklearn.preprocessing import StandardScaler


# In[41]:


#One hot encode
encoded_col = ['Course_Domain', 'Course_Type']


# In[43]:


traintest = pd.get_dummies(traintest, columns = encoded_col)


# In[48]:


traintest.info()


# In[49]:


traintest.columns


# In[52]:


scaler = StandardScaler()
a = pd.DataFrame(scaler.fit_transform(traintest[[ 'Day_No', 'Course_ID', 'Short_Promotion', 'Public_Holiday',
       'Long_Promotion', 'Competition_Metric',
       'Course_Domain_Business', 'Course_Domain_Development',
       'Course_Domain_Finance & Accounting',
       'Course_Domain_Software Marketing', 'Course_Type_Course',
       'Course_Type_Degree', 'Course_Type_Program']]), columns = [ 'Day_No', 'Course_ID', 'Short_Promotion', 'Public_Holiday',
       'Long_Promotion', 'Competition_Metric', 
       'Course_Domain_Business', 'Course_Domain_Development',
       'Course_Domain_Finance & Accounting',
       'Course_Domain_Software Marketing', 'Course_Type_Course',
       'Course_Type_Degree', 'Course_Type_Program'])


# In[70]:


traintest['ID']


# In[76]:


train['Competition_Metric'].fillna(train['Competition_Metric'].mean(),inplace = True)


# In[78]:


train.drop('User_Traffic', axis = 1, inplace = True)


# In[79]:


train.info()


# In[144]:


train = pd.get_dummies(train,columns = encoded_col)


# In[82]:


train.columns


# In[92]:


train_scaled = pd.DataFrame(scaler.fit_transform(train[['Day_No', 'Course_ID', 'Short_Promotion', 'Public_Holiday',
       'Long_Promotion', 'Competition_Metric', 
       'Course_Domain_Business', 'Course_Domain_Development',
       'Course_Domain_Finance & Accounting',
       'Course_Domain_Software Marketing', 'Course_Type_Course',
       'Course_Type_Degree', 'Course_Type_Program']]), columns = col)


# In[93]:


col=['Day_No', 'Course_ID', 'Short_Promotion', 'Public_Holiday',  'Long_Promotion', 'Competition_Metric','Course_Domain_Business', 'Course_Domain_Development','Course_Domain_Finance & Accounting','Course_Domain_Software Marketing', 'Course_Type_Course', 'Course_Type_Degree', 'Course_Type_Program']


# In[96]:


train_scaled_new = pd.concat([train_scaled,train[['ID','Sales']]], axis = 1)


# In[100]:


from sklearn.model_selection import train_test_split


# In[101]:


X = train_scaled
y = train['Sales']


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3)


# In[ ]:





# In[297]:


# Linear Regression
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)
metrics.r2_score(y_test,y_pred_lr)


# In[298]:


# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
metrics.r2_score(y_test,y_pred_rf)


# In[238]:


from xgboost import XGBRegressor


# In[297]:


# Linear Regression
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)
metrics.r2_score(y_test,y_pred_lr)


# In[298]:


# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
metrics.r2_score(y_test,y_pred_rf)


# In[238]:


from xgboost import XGBRegressor


# In[111]:


model_xg = XGBRegressor()
model_xg.fit(X_train,y_train)
y_pred_xg = model_xg.predict(X_test)


# In[112]:


metrics.r2_score(y_test, y_pred_xg)


# In[113]:


test.head()


# In[114]:


test = pd.get_dummies(test,columns = sub_encode_col)


# In[ ]:


test.drop('ID', axis= 1, inplace = True)


# In[ ]:


test['Competition_Metric'].fillna(test['Competition_Metric'].mean(),inplace = True)


# In[ ]:


test.columns


# In[ ]:


sub_y_pred_lr = model_lr.predict(test)
sub_y_pred_rf = model_rf.predict(test)
sub_y_pred_xg = model_xg.predict(test)


# In[ ]:


submission_lr = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_lr))
submission_rf = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_rf))
submission_xg = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_xg))


# In[ ]:


submission_lr = submission_lr.rename(columns={'ID':'ID',0:'Sales'})
submission_rf = submission_rf.rename(columns={'ID':'ID',0:'Sales'})
submission_xg = submission_xg.rename(columns={'ID':'ID',0:'Sales'})


# In[319]:


metrics.r2_score(y_test, y_pred_xg)


# In[305]:


sub_encode_col = ['Course_Domain','Course_Type']


# In[306]:


test.head()


# In[131]:


test = pd.get_dummies(test,columns = encoded_col)


# In[127]:


test.drop('ID', axis= 1, inplace = True)


# In[128]:


test['Competition_Metric'].fillna(test['Competition_Metric'].mean(),inplace = True)


# In[132]:


test.columns


# In[135]:


X_train.columns


# In[136]:


sub_y_pred_lr = model_lr.predict(test)
sub_y_pred_rf = model_rf.predict(test)
sub_y_pred_xg = model_xg.predict(test)


# In[139]:


submission_lr = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_lr))
submission_rf = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_rf))
submission_xg = pd.DataFrame(submission_id).join(pd.DataFrame(sub_y_pred_xg))


# In[140]:


submission_lr = submission_lr.rename(columns={'ID':'ID',0:'Sales'})
submission_rf = submission_rf.rename(columns={'ID':'ID',0:'Sales'})
submission_xg = submission_xg.rename(columns={'ID':'ID',0:'Sales'})


# In[142]:


submission_lr.to_csv('submission_lr_2.csv')


# In[143]:


submission_rf.to_csv('submission_rf_2.csv')
submission_xg.to_csv('submission_xg_2.csv')


# In[ ]:




