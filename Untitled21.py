#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[6]:


df=pd.read_csv('c:/4-DataSets/Boston.csv',index_col='Unnamed: 0')


# In[7]:


df.head()


# In[8]:


df.describe()


# In[10]:


sum(df.duplicated())


# In[12]:


df.mean()


# In[13]:


df.std()


# In[14]:


df.dtypes


# In[16]:


df.columns


# # crim_columns

# #crim columns having a multiple outlier

# In[17]:
 

sns.boxplot(x=df.crim)


# In[18]:


iqr=df.crim.quantile(0.75) - df.crim.quantile(0.25)


# In[31]:


upper_crim=df.crim.quantile(0.75) + 1.5 * iqr
upper_crim


# In[32]:


lower_crim=df.crim.quantile(0.25) - 1.5 * iqr
lower_crim


# # Trimming

# #removing outlier

# In[24]:


outlier=np.where(df.crim > upper_crim , True , np.where( df.crim < lower_crim ,True ,False))


# In[26]:


sum(outlier)
#there are total 66 outlier in the crim columns


# In[27]:


df_t=df.loc[~outlier]


# In[28]:


df.shape


# In[29]:


df_t.shape


# In[30]:


sns.boxplot(x=df_t.crim)


# # masking

# #converting outlier into lower and  upper limit

# In[35]:


df_m=np.where( df.crim > upper_crim , upper_crim ,  np.where( df.crim < lower_crim , lower_crim , df.crim) )


# In[36]:


sns.boxplot(x=df_m)


# # winsorizer

# In[44]:


from feature_engine.outliers import winsorizer


# In[42]:





# In[ ]:




