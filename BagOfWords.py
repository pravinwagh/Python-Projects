
# coding: utf-8

# In[1]:

#import library
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:

# create CountVectorizer object
vectoriser = CountVectorizer()


# In[3]:

# Sample strings
document1 = 'Hi How are you'
document2 = 'today is a very very very pleasant day and we can have some fun fun fun'
document3 = 'This was an amazing experience'


# In[4]:

# Create an array
listofdocuments = [document1,document2,document3]


# In[5]:

# Fit in CountVectoriser model
bag_of_words = vectoriser.fit(listofdocuments)


# In[6]:

# Verify the result
bag_of_words


# In[7]:

# Tokenise the words
bag_of_words = vectoriser.transform(listofdocuments)


# In[8]:

# View the results
print bag_of_words


# In[9]:

# Search the words using vocabulary_.get function
print vectoriser.vocabulary_.get('very')
print vectoriser.vocabulary_.get('fun')


# In[10]:

# View the datatype
type(bag_of_words)

