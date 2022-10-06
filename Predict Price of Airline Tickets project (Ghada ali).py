#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Import packeges
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn
import datetime 
import graphviz

from pandas import DataFrame
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
#from mlxtend.evaluate import bias_variance_decomp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn import tree
from datetime import date
from datetime import time


# In[5]:


# load data
data = pd.read_csv('Predict Price of Airline Tickets.csv')
data.head()


# # Summary statistics

# In[11]:


data.info()
data.shape


# In[12]:


data.describe(include='all')


# In[6]:


#data['year'] = pd.DatetimeIndex(data['Date_of_Journey']).year
data['month'] = pd.DatetimeIndex(data['Date_of_Journey']).month
data['day'] = pd.DatetimeIndex(data['Date_of_Journey']).day
data


# In[14]:


data.isna().sum()#.sum()


# In[7]:


## Encoder data 

le = LabelEncoder()

data_LE = data.apply(le.fit_transform)

data_LE.head()


# In[16]:


data_LE.describe(include='all')


# So, what do I know about dataset? The Predict Price of Airline Tickets dataset has 11 columns (features) and 10683, rows (observations,samples).
# 
# Numerical columns are:
# 
# Price
# 
# year,month,day of Date_of_Journey had been spreated to 3 colunm to get thier numerical values
# 
# Categorical columns are:
# 
# Total_Stops (5 (0,1,2,3,4))
# 
# Airline (12,(Jet Airways, IndiGo, Air India, Multiple carriers, SpiceJet, Vistara, Air Asia, GoAir, Multiple carriers Premium economy, Jet Airways Business, Vistara Premium economy, Trujet)
# 
# Date_of_Journey (44)
# 
# Route (128)
# 
# Source (5, (Delhi, Kolkata, Banglore, Mumbai, Chennai))
# 
# Destination (6, [Cochin, Banglore, Delhi, New Delhi, Hyderabad, Kolkata])
# Duration (368)

# # EDA

# In[11]:


fig, axes = plt.subplots(1, 1, figsize=(12,8)) 
sns.heatmap(data_LE.corr(), annot=True)


# can be noitce source, destnaion, route and total stop have a high correlation with price  

# In[26]:


sns.pairplot(data_LE)


# In[27]:


sns.pairplot(data)


# In[32]:


#Relationship between 2 Numerical Variables
fig, axes = plt.subplots(2, 2, figsize=(15,12)) # plot 4 graphs  
sns.scatterplot(x='Duration', y='Price', data=data, ax=axes[0,0])
sns.scatterplot(x ='month', y ='Price', data = data, ax=axes[0,1])
sns.scatterplot(x ='day', y ='Price', data = data, ax=axes[1,0])
sns.scatterplot(x = 'Total_Stops', y = 'Price', data = data_LE)
#Airline	Date_of_Journey	Source	Destination	Route	Dep_Time	Arrival_Time	Duration	Total_Stops	Additional_Info	Price	year	month	day


# In[29]:


#Categorical Variables
fig, axes = plt.subplots(2, 3, figsize=(15,12)) # plot 6 graphs
sns.countplot(x='Airline', data=data, ax=axes[0,0]).set_title('Airline')
sns.countplot(x='Source', data=data, ax=axes[0,1]).set_title('Source')
sns.countplot(x='Destination', data=data, ax=axes[1,0]).set_title('Destination')
sns.countplot(x='Route', data=data, ax=axes[1,1]).set_title('Route')
sns.countplot(x='Duration', data=data, ax=axes[0,2]).set_title('Duration')
sns.countplot(x='Total_Stops', data=data, ax=axes[1,2]).set_title('Total_Stops')


# In[19]:


fig, axes = plt.subplots(2, 3, figsize=(15,12)) # plot 6 graphs  
sns.scatterplot(x='Airline', y='Price', data=data_LE, ax=axes[0,0])
sns.scatterplot(x ='Total_Stops', y ='Price', data = data_LE, ax=axes[0,1])
sns.scatterplot(x ='Route', y ='Price', data = data_LE, ax=axes[1,0])
sns.scatterplot(x = 'Destination', y = 'Price', data = data_LE, ax=axes[1,1])
sns.scatterplot(x = 'Source', y = 'Price', data = data_LE, ax=axes[1,2]) 
sns.scatterplot(x = 'Duration', y = 'Price', data = data_LE, ax=axes[0,2]) 


# In[20]:


fig, axes = plt.subplots(2, 3, figsize=(15,12)) # plot 6 graphs  
sns.scatterplot(x='Airline', y='Duration', data=data_LE, ax=axes[0,0])
sns.scatterplot(x ='Total_Stops', y ='Route', data = data_LE, ax=axes[0,1])
sns.scatterplot(x ='Route', y ='Source', data = data_LE, ax=axes[1,0])
sns.scatterplot(x = 'Destination', y = 'Source', data = data_LE, ax=axes[1,1])
sns.scatterplot(x = 'Route', y = 'Destination', data = data_LE, ax=axes[1,2]) 
sns.scatterplot(x = 'Date_of_Journey', y = 'Dep_Time', data = data_LE, ax=axes[0,2]) 


# In[21]:


fig, axes = plt.subplots(2, 3, figsize=(15,12)) # plot 6 graphs  
sns.scatterplot(x='Airline', y='Total_Stops', data=data_LE, ax=axes[0,0])
sns.scatterplot(x ='Date_of_Journey', y ='Total_Stops', data = data_LE, ax=axes[0,1])
sns.scatterplot(x ='Route', y ='Total_Stops', data = data_LE, ax=axes[1,0])
sns.scatterplot(x = 'Destination', y = 'Total_Stops', data = data_LE, ax=axes[1,1])
sns.scatterplot(x = 'Source', y = 'Total_Stops', data = data_LE, ax=axes[1,2]) 
sns.scatterplot(x = 'Duration', y = 'Total_Stops', data = data_LE, ax=axes[0,2]) 


# In[28]:


fig, axes = plt.subplots(2, 3, figsize=(20,12)) # plot 6 graphs 
sns.boxplot(data=data_LE, x='Total_Stops', y='Airline', ax=axes[0][0])
sns.boxplot(data=data_LE, x='Total_Stops', y='Date_of_Journey', ax=axes[0][1])
sns.boxplot(data=data_LE, x='Total_Stops', y='Route', ax=axes[0][2])
sns.boxplot(data=data_LE, x='Total_Stops', y='Destination', ax=axes[1][0])
sns.boxplot(data=data_LE, x='Total_Stops', y='Source', ax=axes[1][1])
sns.boxplot(data=data_LE, x='Total_Stops', y='Duration', ax=axes[1][2])


# In[29]:


fig, axes = plt.subplots(2, 3, figsize=(20,12)) # plot 6 graphs 
sns.boxplot(data=data_LE, x='Additional_Info', y='Airline', ax=axes[0][0])
sns.boxplot(data=data_LE, x='Additional_Info', y='Date_of_Journey', ax=axes[0][1])
sns.boxplot(data=data_LE, x='Additional_Info', y='Route', ax=axes[0][2])
sns.boxplot(data=data_LE, x='Additional_Info', y='Destination', ax=axes[1][0])
sns.boxplot(data=data_LE, x='Additional_Info', y='Source', ax=axes[1][1])
sns.boxplot(data=data_LE, x='Additional_Info', y='Duration', ax=axes[1][2])


# In[22]:


# train_test_split for data
x = data_LE.drop(['Price'],axis=1)
y = data_LE['Price']

x_train,x_test, y_train,y_test  = train_test_split(x,y, test_size=0.33, random_state=42)


# #  Decision Tree model 

# In[23]:


D_T_regressor = DecisionTreeRegressor()

D_T_regressor.fit(x_train,y_train)


# In[24]:


columns_name = x.columns.tolist()

print(tree.export_text(D_T_regressor,feature_names=columns_name))


# In[8]:


tree_fig = tree.export_graphviz(D_T_regressor,feature_names=columns_name, filled=True)

graph = graphviz.Source(tree_fig, format='png')

graph


# In[25]:


y_pred = D_T_regressor.predict(x_test)

print('MSE = ', metrics.mean_squared_error(y_test,y_pred))
print('MAE = ', metrics.mean_absolute_error(y_test,y_pred))


# In[26]:


# Importance feature in model
importance = D_T_regressor.feature_importances_

indices = np.argsort(importance)

plt.barh(range(len(indices)), importance[indices], color='b')

plt.yticks(range(len(indices)), [columns_name[i] for i in indices])


# # Random forest Regressor Model

# In[27]:


regressor = RandomForestRegressor()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


# In[28]:


MAE  = metrics.mean_absolute_error(y_test,y_pred)
MAPE = metrics.mean_absolute_percentage_error(y_test,y_pred)
MSE = metrics.mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
R2_SCORE = metrics.r2_score(y_test,y_pred)

print("MAE = ",MAE)
print("MAPE = ",MAPE)
print("MSE = ",MSE)
print("RMSE = ",RMSE)
print("R2_SCORE = ",R2_SCORE)


# In[29]:


np.mean(y_test)


# In[ ]:


for tree_in_forest in regressor.estimators_:
    dot_data = tree.export_graphviz(tree_in_forest, feature_names=x_train.columns.tolist(), out_file=None, filled=True, rounded=True)
    graph = graphviz.Source(dot_data,format='png')
    graph
    
graph.render('Trees', format='png', view=True)


# In[49]:


features = x_train.columns.tolist()
importances = regressor.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# # cross validation

# In[51]:


cv_conf = KFold(n_splits= 5, random_state=1, shuffle=True)

scores = cross_val_score(regressor, x, y, cv = cv_conf, scoring='neg_mean_squared_error' )

final_score = np.mean(np.absolute(scores))

print('mean squared Error = ', final_score)
# @ n_splits= 5 mean squared Error=


# In[ ]:




