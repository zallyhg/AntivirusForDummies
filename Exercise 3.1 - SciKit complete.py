
# coding: utf-8

# In[1]:


import pandas
from sklearn import ensemble
from sklearn import metrics


# For starters, only load 1000 malicious and benign data sets (each)

# In[2]:


READ_LINES = 1000
malicious_data = pandas.read_csv('rodata/clam_mal.csv', index_col=None, nrows=READ_LINES)
benign_data = pandas.read_csv('rodata/clam_benign.csv', index_col=None, nrows=READ_LINES)


# Merge malicious and clean data (features), remove sample hashes, and split out classes

# In[3]:


learn_data = malicious_data.append(benign_data, ignore_index=True)
learn_data.fillna(0, inplace=True) # fill missing entries from either set with 0
classes = learn_data['malicious']
del learn_data['malicious']
del learn_data['hash']


# Q: How many features does the data set have? 
# 
# Q: If every data point consumes 1k memory, how much memory does the data consume now?
# 
# Q: What kind of features do you see? Anything you find strange / would expect differently?

# In[4]:


pandas.set_option('display.max_columns', 200)
learn_data.head(10)


# We will use a Random Forest (multiple Decision Trees combined)
# 
# Experiment 1: Re-run the training & result several times. Does the result change? How much does it vary?

# In[6]:


classifier = ensemble.RandomForestClassifier()


# Tell the classifier to learn based on reference data

# In[7]:


classifier.fit(learn_data, classes)


# How well does the classifier work on the original data?

# In[ ]:


prediction = classifier.predict(learn_data)
print metrics.accuracy_score(classes, prediction)
print metrics.confusion_matrix(classes, prediction)

