#!/usr/bin/env python
# coding: utf-8

# In[32]:


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


# In[3]:


house_price = pd.read_csv("house price.csv")
house_price


# In[5]:


house_price.info()


# In[6]:


house_price.describe()


# In[8]:


fig, axes = plt.subplots(1, 1, figsize=(12,8)) # plot 6 graphs 
sns.heatmap(house_price[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station',
                'X4 number of convenience stores', 'X5 latitude','X6 longitude',
                'Y house price of unit area']].corr(), annot=True)


# In[10]:


sns.pairplot(house_price)


# In[22]:


#### multi linear Regression
x = house_price[['X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores']].values
y = house_price['Y house price of unit area'].values
np.mean(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=215  )
#x_test
model = linear_model.LinearRegression()
model.fit(x_train,y_train)

model.coef_

model.predict([[12,500,3]])
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


# In[38]:


## ploynomial regression
x = house_price[['X3 distance to the nearest MRT station']].values
y = house_price['Y house price of unit area'].values
np.mean(y)
x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1),y, test_size= 0.33, random_state=215  )
x_test


model = linear_model.LinearRegression()

model.fit(x_train,y_train)


avg_loss, avg_bais, avg_var = bias_variance_decomp(model,x_train,y_train,x_test,y_test, loss = 'mse', random_seed=133  )


#print('Avarage Loss MSE = ',avg_loss )
#print('Avarage bais = ',avg_bais )
#print('Avarage variance = ',avg_var )

## degree of polynomial
poly_reg = PolynomialFeatures(degree=4)
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
print('Avarage variance = ',avg_var )


# linear regression result
# Avarage Loss MSE =  126.58405119465826
# Avarage bais =  125.86125912570397
# Avarage variance =  0.722792068954264
#  
# polynomial regression result 
# 
# Degree 2 is the optiumum degree
# Avarage Loss MSE =  116.4796291121976
# Avarage bais =  115.18003129021302
# Avarage variance =  1.2995978219845898
# 
# Degree 3
# Avarage Loss MSE =  116.67272446098916
# Avarage bais =  109.61343373927478
# Avarage variance =  7.059290721714412
# 
# Degree 4
# Avarage Loss MSE =  145.0385736655714
# Avarage bais =  107.74722541449601
# Avarage variance =  37.2913482510754
# 

# In[ ]:




