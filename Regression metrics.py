#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#Packages for various types of regressions
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[2]:


tortoises = pd.read_csv('Tortoises.csv')

#Store relevant columns as variables
X = tortoises[['Length']].values.reshape(-1, 1)
y = tortoises[['Clutch']].values.reshape(-1, 1)


# In[3]:


#Fit a quadratic regression model
polyFeatures = PolynomialFeatures(degree=2, include_bias=False)
xPoly = polyFeatures.fit_transform(X)
polyModel = LinearRegression()
polyModel.fit(xPoly, y)
yPolyPredicted = polyModel.predict(xPoly)

#graph the quadratic regression
plt.scatter(X,y,color='black')
xDelta = np.linspace(X.min(),X.max(),1000)
yDelta = polyModel.predict(polyFeatures.fit_transform(xDelta.reshape(-1,1)))
plt.plot(xDelta, yDelta,color = 'blue',linewidth = 2)
plt.xlabel('Length (mm)',fontsize=14)
plt.ylabel('Clutch size',fontsize=14)
plt.ylim(0,16);


# In[4]:


r2_score(y,yPolyPredicted)


# In[5]:


mean_squared_error(y,yPolyPredicted)


# In[6]:


mean_squared_error(y,yPolyPredicted,squared=False)


# In[7]:


mean_absolute_error(y,yPolyPredicted)


# In[8]:


#Fit a linear model
linModel = LinearRegression()
linModel.fit(X,y)
yLinPredicted = linModel.predict(X)

#graph the linear regression
plt.scatter(X,y,color='black')
xDelta = np.linspace(X.min(),X.max(),1000)
yDelta = linModel.predict(xDelta.reshape(-1,1))
plt.plot(xDelta,yDelta,color='blue',linewidth = 2)
plt.xlabel('Length (mm)',fontsize=14)
plt.ylabel('Clutch size',fontsize=14)
plt.ylim(0,16);

#print the scores
print('R-squared = ',r2_score(y,yLinPredicted))
print('MSE = ',mean_squared_error(y, yLinPredicted))
print('RMSE = ',mean_squared_error(y, yLinPredicted,squared=False))
print('MAE = ',mean_absolute_error(y, yLinPredicted))


# In[9]:


#Add in an usual observation at (283,15)
X2 = np.append(X,283)
X2 = X2.reshape(-1, 1)
y2 = np.append(y,15)
y2= y2.reshape(-1, 1)

#Fit a new quadratic regression model
polyFeatures2 = PolynomialFeatures(degree=2, include_bias=False)
xPoly2 = polyFeatures.fit_transform(X2)
polyModel2 = LinearRegression()
polyModel2.fit(xPoly2, y2)
yPolyPredicted2 = polyModel2.predict(xPoly2)

#graph the new quadratic regression with the old regression opaque
plt.scatter(X,y,color='black')
plt.scatter(283,15,marker='*',color='orange',s=150)
xDelta = np.linspace(X.min(),X.max(),1000)
yDelta = polyModel.predict(polyFeatures.fit_transform(xDelta.reshape(-1,1)))
plt.plot(xDelta,yDelta,color='blue',linewidth=2,alpha=0.2)
yDelta2 = polyModel2.predict(polyFeatures2.fit_transform(xDelta.reshape(-1,1)))
plt.plot(xDelta,yDelta2,color='blue',linewidth=2)
plt.xlabel('Length (mm)',fontsize=14)
plt.ylabel('Clutch size',fontsize=14)
plt.ylim(0,16)
plt.xlim(281.5,336.5)

print('R-squared = ',r2_score(y2,yPolyPredicted2))
print('MSE = ',mean_squared_error(y2,yPolyPredicted2))
print('RMSE = ',mean_squared_error(y2,yPolyPredicted2,squared=False))
print('MAE = ',mean_absolute_error(y2,yPolyPredicted2))


# In[10]:


#Fit a new linear model
linModel2 = LinearRegression()
linModel2.fit(X2,y2)
yLinPredicted2 = linModel.predict(X2)

#graph the new linear regression with the old regression opaque
plt.scatter(X,y,color='black')
plt.scatter(283,15,marker='*',color='orange',s=150)
xDelta = np.linspace(X.min(),X.max(),1000)
yDelta = linModel.predict(xDelta.reshape(-1,1))
plt.plot(xDelta,yDelta,color='blue',linewidth = 2,alpha=0.2)
yDelta2 = linModel2.predict(xDelta.reshape(-1,1))
plt.plot(xDelta,yDelta2,color='blue',linewidth = 2)
plt.xlabel('Length (mm)',fontsize=14)
plt.ylabel('Clutch size',fontsize=14)
plt.ylim(0,16)
plt.xlim(281.5,336.5)

#print the scores
print('R-squared = ',r2_score(y2,yLinPredicted2))
print('MSE = ',mean_squared_error(y2,yLinPredicted2))
print('RMSE = ',mean_squared_error(y2,yLinPredicted2,squared=False))
print('MAE = ',mean_absolute_error(y2,yLinPredicted2))


# In[ ]:




