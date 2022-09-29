#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import packeges
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from pandas import DataFrame
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


#load Data
FuelConsumption = pd.read_csv("FuelConsumptionCo2.csv")
FuelConsumption.head()


# In[56]:


FuelConsumption.info


# In[5]:


FuelConsumption.isna().sum()#.sum()


# In[6]:


# Summary statistics
#FuelConsumption.describe() # only for numerical variables 
FuelConsumption.describe(include='all') # for all variables


# In[8]:


FuelConsumption.corr()


# ####  So, what do I know about dataset?
# 
# - The *FuelConsumption* dataset has 13 columns (features) and 1067 rows (observations,samples).\
#   *Numerical* columns are:
#     - *FUELCONSUMPTION_CITY* 
#     - *FUELCONSUMPTION_HWY* in the high way
#     - *FUELCONSUMPTION__COMB* in both city and high way
#     - *FUELCONSUMPTION__COMB_MPG* 
# 
#   *Categorical* columns are:
#     - *FUELTYPE* (Z,D,X,E) 
#     - *MODEL* 
#     - *VEHICLECLASS* 
#     
# - There is no change in year (2014)
# - There is no missing values.

# EDA

# In[40]:


MODELYEAR=FuelConsumption['MODELYEAR']
MODEL=FuelConsumption['MODEL']
VEHICLECLASS=FuelConsumption['VEHICLECLASS']
ENGINESIZE=FuelConsumption['ENGINESIZE']
CYLINDERS=FuelConsumption['CYLINDERS']
TRANSMISSION=FuelConsumption['TRANSMISSION']
FUELTYPE=FuelConsumption['FUELTYPE']
FUELCONSUMPTION_CITY=FuelConsumption['FUELCONSUMPTION_CITY']
FUELCONSUMPTION_HWY=FuelConsumption['FUELCONSUMPTION_HWY']
FUELCONSUMPTION_COMB_MPG=FuelConsumption['FUELCONSUMPTION_COMB_MPG']
CO2EMISSIONS=FuelConsumption['CO2EMISSIONS']


# In[24]:


#detect labels in categorical variables
for col in FuelConsumption.columns[2:13]:
    print(col, np.unique(FuelConsumption[col]))


# In[15]:


##Visualization
#Categorical Variables
fig, axes = plt.subplots(2, 2, figsize=(15,12)) # plot 4 graphs
sns.countplot(x='MODEL', data=FuelConsumption, ax=axes[0,0]).set_title('MODEL Types')
sns.countplot(x='VEHICLECLASS', data=FuelConsumption, ax=axes[0,1]).set_title('VEHICLECLASS Types')
sns.countplot(x='TRANSMISSION', data=FuelConsumption, ax=axes[1,0]).set_title('TRANSMISSION Types')
sns.countplot(x='FUELTYPE', data=FuelConsumption, ax=axes[1,1]).set_title('FUEL Types')


# In[18]:


#Visualization
#Relationship between 2 Numerical Variables
fig, axes = plt.subplots(2, 2, figsize=(15,12)) # plot 4 graphs 
# simple scatter plot between two variables 
sns.scatterplot(x='FUELCONSUMPTION_CITY', y='CO2EMISSIONS', data=FuelConsumption, ax=axes[0,0])
# group by time and show the groups with different colors
sns.scatterplot(x ='FUELCONSUMPTION_HWY', y ='CO2EMISSIONS', data = FuelConsumption, ax=axes[0,1])
# variable time by varying both color and marker
sns.scatterplot(x ='FUELCONSUMPTION_COMB_MPG', y ='CO2EMISSIONS', data = FuelConsumption, ax=axes[1,0])
# vary colors and markers to show two different grouping variables
sns.scatterplot(x = 'FUELCONSUMPTION_CITY', y = 'CO2EMISSIONS', data = FuelConsumption)


# In[ ]:


#Visualization
#Relationship between 2 Numerical Variables
fig, axes = plt.subplots(2, 2, figsize=(15,12)) # plot 4 graphs 

# simple scatter plot between two variables 
sns.scatterplot(x='FUELCONSUMPTION_CITY', y='CO2EMISSIONS', data=FuelConsumption, ax=axes[0,0])

# group by time and show the groups with different colors
sns.scatterplot(x ='FUELCONSUMPTION_HWY', y ='CO2EMISSIONS', data = FuelConsumption, ax=axes[0,1])

# variable time by varying both color and marker
sns.scatterplot(x ='FUELCONSUMPTION_COMB_MPG', y ='CO2EMISSIONS', data = FuelConsumption, ax=axes[1,0])

# vary colors and markers to show two different grouping variables
sns.scatterplot(x = 'FUELCONSUMPTION_CITY', y = 'CO2EMISSIONS', data = FuelConsumption)


# In[19]:


sns.pairplot(FuelConsumption)


# In[20]:


# fit linear regression models to the scatter plots, show density plots on diagonal
sns.pairplot(FuelConsumption, kind='reg', diag_kind="kde")


# In[21]:


fig, axes = plt.subplots(1, 1, figsize=(12,8)) # plot 6 graphs 
sns.heatmap(FuelConsumption[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
                'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG',
                'CO2EMISSIONS']].corr(), annot=True)


# In[19]:


fig, axes = plt.subplots(2, 2, figsize=(25,13)) # plot 4 graphs 
sns.scatterplot(data=FuelConsumption, x='FUELCONSUMPTION_CITY', y='CO2EMISSIONS',hue='FUELTYPE', ax=axes[0][0])
sns.scatterplot(data=FuelConsumption, x='FUELCONSUMPTION_HWY', y='CO2EMISSIONS', hue='FUELTYPE', ax=axes[0][1])
sns.scatterplot(data=FuelConsumption, x='FUELCONSUMPTION_COMB', y='CO2EMISSIONS', hue='FUELTYPE', ax=axes[1][0])
sns.scatterplot(data=FuelConsumption, x='FUELCONSUMPTION_COMB_MPG', y='CO2EMISSIONS', hue='FUELTYPE',ax=axes[1][1])


# Regression
# linear regression:
# 1- ENGINESIZE acorrding to FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY, FUELCONSUMPTION_COMB_MPG, CO2EMISSIONS 
# multiregression :
# 1- CO2EMISSIONS  acorrding to FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY,FUELCONSUMPTION_COMP RESPECT TO FUELTYPE
# bolynomial regression
# 1-CO2EMISSIONS and engisize  acorrding to FUELCONSUMPTION_COMB_MPG

# In[33]:


### linear Regression ENGINESIZE acorrding to FUELCONSUMPTION_CITY
# set x & y labeles
x = FuelConsumption[['FUELCONSUMPTION_CITY']].values
y = FuelConsumption['ENGINESIZE'].values

model = linear_model.LinearRegression()
model.fit(x,y)

model.coef_
x_test = [[2]]

model.predict([[12]])
#print(model)

from sklearn.metrics import mean_squared_error , r2_score
# y_pred & y_true
y_pred =model.predict(x)
MSE =mean_squared_error(y,y_pred)
r_score = r2_score(y,y_pred)
print('mean_squared_value = ',MSE)
print('r2_score_value = ',r_score)
a_0 = model.intercept_
a_1 = model.coef_
print('a_0 =',a_0 )
print('a_1 =',a_1 )

plt.scatter(x,y)
plt.plot(x,y_pred,c='r')

import statsmodels.api as sm
x=sm.add_constant(x)
sm_model = sm.OLS(y,x).fit()
predection = sm_model.predict(x)

sm_model_summary = sm_model.summary()
print(sm_model_summary)


# In[34]:


### linear Regression ENGINESIZE acorrding to FUELCONSUMPTION_HWY
# set x & y labeles
x = FuelConsumption[['FUELCONSUMPTION_HWY']].values
y = FuelConsumption['ENGINESIZE'].values

model = linear_model.LinearRegression()
model.fit(x,y)

model.coef_
x_test = [[2]]

model.predict([[12]])
#print(model)

from sklearn.metrics import mean_squared_error , r2_score
# y_pred & y_true
y_pred =model.predict(x)
MSE =mean_squared_error(y,y_pred)
r_score = r2_score(y,y_pred)
print('mean_squared_value = ',MSE)
print('r2_score_value = ',r_score)
a_0 = model.intercept_
a_1 = model.coef_
print('a_0 =',a_0 )
print('a_1 =',a_1 )

plt.scatter(x,y)
plt.plot(x,y_pred,c='r')

import statsmodels.api as sm
x=sm.add_constant(x)
sm_model = sm.OLS(y,x).fit()
predection = sm_model.predict(x)

sm_model_summary = sm_model.summary()
print(sm_model_summary)


# In[35]:


### linear Regression ENGINESIZE acorrding to FUELCONSUMPTION_COMB
# set x & y labeles
x = FuelConsumption[['FUELCONSUMPTION_COMB']].values
y = FuelConsumption['ENGINESIZE'].values

model = linear_model.LinearRegression()
model.fit(x,y)

model.coef_
x_test = [[2]]

model.predict([[12]])
#print(model)

from sklearn.metrics import mean_squared_error , r2_score
# y_pred & y_true
y_pred =model.predict(x)
MSE =mean_squared_error(y,y_pred)
r_score = r2_score(y,y_pred)
print('mean_squared_value = ',MSE)
print('r2_score_value = ',r_score)
a_0 = model.intercept_
a_1 = model.coef_
print('a_0 =',a_0 )
print('a_1 =',a_1 )

plt.scatter(x,y)
plt.plot(x,y_pred,c='r')

import statsmodels.api as sm
x=sm.add_constant(x)
sm_model = sm.OLS(y,x).fit()
predection = sm_model.predict(x)

sm_model_summary = sm_model.summary()
print(sm_model_summary)


# In[8]:


#### multi linear Regression

x = FuelConsumption[['FUELCONSUMPTION_COMB','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB_MPG']].values
y = FuelConsumption['CO2EMISSIONS'].values
np.mean(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=215  )
#x_test
model = linear_model.LinearRegression()
model.fit(x_train,y_train)

model.coef_

#model.predict([[12,500,3]])
# predicting the accuracy score
y_pred = model.predict(x_test)

r_score = r2_score(y_test,y_pred)
MSE = mean_squared_error(y_test,y_pred)

RMSE = np.sqrt(mean_squared_error(y_test,y_pred))

print('‘r2 socre is = ',r_score)
print('‘mean_sqrd_error is =',MSE )
print('‘root_mean_squared error of is =',RMSE)

a_0 = model.intercept_
a_1 = model.coef_

print('a_0 is = ',a_0)
print('a_1 is =',a_1 )

model = linear_model.LinearRegression()

model.fit(x_train,y_train)


avg_loss, avg_bais, avg_var = bias_variance_decomp(model,x_train,y_train,x_test,y_test, loss = 'mse', random_seed=133  )


print('Avarage Loss MSE = ',avg_loss )
print('Avarage bais = ',avg_bais )
print('Avarage variance = ',avg_var )

x = sm.add_constant(x) # adding a constant

model = sm.OLS(y, x).fit()

predictions = model.predict(x) 

print_model = model.summary()
print(print_model)


# In[16]:


## ploynomial regression
x = FuelConsumption[['FUELCONSUMPTION_COMB_MPG']].values
y = FuelConsumption['CO2EMISSIONS'].values
np.mean(y)
x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1),y, test_size= 0.33, random_state=215  )
x_test


model = linear_model.LinearRegression()

model.fit(x_train,y_train)


avg_loss, avg_bais, avg_var = bias_variance_decomp(model,x_train,y_train,x_test,y_test, loss = 'mse', random_seed=133  )


print('Avarage Loss MSE = ',avg_loss )
print('Avarage bais = ',avg_bais )
print('Avarage variance = ',avg_var )

## degree of polynomial
poly_reg = PolynomialFeatures(degree=6)
X_poly = poly_reg.fit_transform(x.reshape(-1, 1))

# split data to train and test set

x_train, x_test, y_train, y_test = train_test_split(X_poly,y, test_size= 0.33, random_state=215  )

lin_reg = LinearRegression()
lin_reg.fit(x,y)


lin_reg2 = LinearRegression()
lin_reg2.fit(x_train,y_train)
 

X_grid = np.arange(min(x),max(x),0.1)

X_grid = X_grid.reshape(len(X_grid),1) 

plt.scatter(x,y, color='red') 
 

# plt.plot(X_grid, poly_reg.fit_transform(X_grid),color='blue') 
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue') 

plt.title("Truth or Bluff(Polynomial)")
plt.xlabel('X distance to the nearest MRT station')
plt.ylabel('Y house price of unit area')
plt.show()


avg_loss, avg_bais, avg_var = bias_variance_decomp(lin_reg2,x_train,y_train,x_test,y_test, loss = 'mse', random_seed=2  )


print('Avarage Loss MSE = ',avg_loss )
print('Avarage bais = ',avg_bais )
print('Avarage variance = ',avg_var)


# linear regression result
# Avarage Loss MSE =  793.0091651848716
# Avarage bais =  789.4351491449297
# Avarage variance =  3.5740160399419527
#  
# polynomial regression result 
# 
# Degree 2        
# Avarage Loss MSE =  659.9581563869814
# Avarage bais =  656.5144401223035
# Avarage variance =  3.4437162646776653
# 
# Degree 3
# Avarage Loss MSE =  663.623659285869
# Avarage bais =  653.8730050751191
# Avarage variance =  9.750654210750318
# 
# Degree 4  is the optiumum degree
# Avarage Loss MSE =  651.1241652952936
# Avarage bais =  629.5865867458184
# Avarage variance =  21.537578549474837
# 
# Degree 5
# Avarage Loss MSE =  653.8694347443742
# Avarage bais =  623.6911663840386
# Avarage variance =  30.178268360334968
# 
# Degree 6
# Avarage Loss MSE =  854.219943849332
# Avarage bais =  634.087458080416
# Avarage variance =  220.1324857689165

# In[ ]:




