#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.polynomial.polynomial import polyfit
from sklearn.linear_model import LinearRegression


# ## __1 - Business Problem__  
# ___Delivery_time -> Predict delivery time using sorting time___

# ## __2 - Data collection and description__ 

# In[3]:


df = pd.read_csv("delivery_time.csv")
df


# ### __Scatter Plot__

# In[3]:


x = df['Sorting Time']
y = df['Delivery Time']


# In[4]:


b, m = polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')
plt.title('Scatter plot Delivery Time')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


# As displayed in the scatter plot, the data does contains some outliers, but there is potive correlation between delivery time and sorting Time

# ### __Correlation Analysis__

# In[5]:


corr = np.corrcoef(x, y)


# Corr  
# array([[1.        , 0.82599726],
#        [0.82599726, 1.        ]])
# 
# The correlation between delivery time and sorting Time is high (83%)

# ## __3 - Regression Model__ 

# ### __1 - No transformation__ 

# In[6]:


model = sm.OLS(y, x).fit()
predictions = model.predict(x)


# In[7]:


model.summary()


# ### __2 - Log Transformation of X__ 

# In[8]:


x_log = np.log(df['Sorting Time'])


# In[9]:


model = sm.OLS(y, x_log).fit()
predictions = model.predict(x_log)


# In[10]:


model.summary()


# ### __3 - Log Transformation of Y__ 

# In[11]:


y_log = np.log(df['Delivery Time'])


# In[12]:


model = sm.OLS(y_log, x).fit()
predictions = model.predict(x)


# In[13]:


model.summary()


# ### __4 - Log Transformation of X & Y__ 

# In[14]:


model = sm.OLS(y_log, x_log).fit()
predictions = model.predict(x_log)


# In[15]:


model.summary()


# ### __5 - Sq Root Transformation of X__ 

# In[16]:


x_sqrt = np.sqrt(df['Sorting Time'])


# In[17]:


model = sm.OLS(y, x_sqrt).fit()
predictions = model.predict(x_sqrt)


# In[18]:


model.summary()


# ### __6 - Square Root Transformation of Y__ 

# In[19]:


y_sqrt = np.sqrt(df['Delivery Time'])


# In[20]:


model = sm.OLS(y_sqrt, x).fit()
predictions = model.predict(x)


# In[21]:


model.summary()


# ### __7 - Square Root Transformation of X & Y__ 

# In[22]:


model = sm.OLS(y_sqrt, x_sqrt).fit()
predictions = model.predict(x_sqrt)


# In[23]:


model.summary()


# ## __4 - Output Interpretation__ 

# We will use Model 7 as it has the best R square value  
# 
# 1 - p-value < 0.01  
# Thus the model is accepted
# 
# 2 - coefficient == 1.64  
# Thus if the value of Sorting Time is increased by 1, the predicted value of Delivery Time will increase by 1.64
# 
# 3 - Adj. R-sqared == 0.987  
# Thus the model explains 98.7% of the variance in dependent variable

# In[ ]:




