#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
import matplotlib.pyplot as py


# In[5]:


X= pd.read_csv('Salary.csv')
X


# In[6]:


print(X.T)


# In[7]:


#Mean centering the Data
X_meaned = X - np.mean(X , axis = 0)
print("X_Meaned")
print(X_meaned)


# In[8]:


# Calculating the covariance matrix of the Mean-Centered Data.
cov_mat = np.cov(X_meaned , rowvar = False)
print("Covariance Matric")
print(cov_mat)


# In[9]:


#Calculating Eigenvalues and Eigenvectors of the covariance matrix
eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
print("Eigen_Vectors")
print(eigen_vectors)


# In[10]:


#sort the eigenvalues in descending order
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
sorted_eigenvalue


# In[11]:


#similarly sort the eigenvectors 
sorted_eigenvectors = eigen_vectors[:,sorted_index]
sorted_eigenvectors


# In[12]:


n_components = 2 #you can select any number of components.
eigenvector_subset = sorted_eigenvectors[:,0:n_components]
print("Eigen Subset")
print(eigenvector_subset)


# In[13]:


#Transform the data 
X_reduced = np.dot(eigenvector_subset.transpose(),X_meaned.transpose()).transpose()
print("Reduced Data")
print(X_reduced)


# In[14]:


#Reconstruction of data
X_reconstrcted=np.dot(X_reduced,eigenvector_subset.transpose())
print("Reconstructed Data")
print(X_reconstrcted)


# In[15]:


# Mean Squared Error
MSE = np.square(np.subtract(X_meaned,X_reconstrcted)).mean()
MSE

