
# coding: utf-8

# In[1]:

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


# In[2]:

estimator = [('dim_reduction',PCA()),('logres_model',LogisticRegression()),('linear_model',LinearRegression())]


# In[3]:

pipeline_estimator = Pipeline(estimator)


# In[4]:

pipeline_estimator


# In[5]:

pipeline_estimator.steps[0]


# In[6]:

pipeline_estimator.steps[1]


# In[7]:

pipeline_estimator.steps

