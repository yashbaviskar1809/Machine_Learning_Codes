#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df=pd.read_csv('Salary_Data.csv')
df.info()


# In[7]:


x=df.iloc[:,[0]].values
y=df.iloc[:,1].values


# In[10]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
LinearRegression()
y_pred-model.predict(x)
y_pred


# In[17]:


import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,y_pred,c='r',label='Best fit Line')
plt.title("LINEAR REGRESSION")
plt.xlabel("EXPERIENCE")
plt.ylabel("SALARY")
plt.legend()

