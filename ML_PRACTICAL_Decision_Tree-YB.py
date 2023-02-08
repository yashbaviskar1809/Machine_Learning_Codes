#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import tree


# In[6]:


#Loading datasets
iris_data = load_iris() 
iris=pd.DataFrame(iris_data.data)


#priting features name of iris data 
print ("Features Name : ", iris_data.feature_names) 

#shape of datasets 
print ("Dataset Shape: ", iris.shape) 

#first five sample 
print ("Dataset: ",iris.head())  


# In[14]:


#priting samples and target 
X = iris.values[:, 0:4] 
print("X =", X)


# In[15]:


Y = iris_data.target
print("Y =", Y)


# In[17]:


# Splitting the dataset into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)


# In[19]:


# Decision tree classifier 
clf= DecisionTreeClassifier(random_state = 100)

#fitting the training data
clf.fit(X_train, y_train)


# In[21]:


# prediction on random data
X=[[6.4,1.8 ,6.6 ,2.1]]
Y_pred=clf.predict(X)
print(Y_pred)

# prediction on X_test (testing data )
Y_pred=clf.predict(X_test)
print(Y_pred)


# In[29]:


from sklearn.metrics import  confusion_matrix
cm=np.array(confusion_matrix(y_test,Y_pred))
cm


# In[25]:


#Tree plotting 
tree.plot_tree(clf)


# In[26]:


#Decision making in decision tree
text_representation = tree.export_text(clf)
print(text_representation)

