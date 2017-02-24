
# coding: utf-8

# In[1]:

#Import Libraries
import pandas as pd
import string
from nltk.corpus import stopwords


# In[2]:

# Load the spam data file
spam_data = pd.read_csv('C:\Users\pravinw\Documents\Fractal\DATA SCIENTIST\PractiseAssignment_Simplilearn\SpamCollection\SpamCollection', sep='\t', names=['Response','Message'])


# In[3]:

# View top 5 records
spam_data.head()


# In[4]:

# Describe the dataset
spam_data.describe()


# In[5]:

# Group by Response
spam_data.groupby('Response').describe()


# In[6]:

# Check the length of each message
spam_data['Length'] = spam_data['Message'].apply(len)


# In[7]:

spam_data.head()


# In[8]:

# Define function to filter out punctuations and stop words
def msg_text_process(msg):
    no_punctuation = [char for char in msg if char not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)
    return[word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]


# In[9]:

# View the output
spam_data['Message'].head(5).apply(msg_text_process)


# In[10]:

# import library for text processing
from sklearn.feature_extraction.text import CountVectorizer


# In[11]:

# use bag of words by applying the function and fit the data into it
bagofwords = CountVectorizer(analyzer=msg_text_process).fit(spam_data['Message'])


# In[12]:

# print length of bag of words stored in vocabulary_
print len(bagofwords.vocabulary_)


# In[13]:

msg_bagofwords = bagofwords.transform(spam_data['Message'])
print msg_bagofwords


# In[14]:

# Import Libraries for tfidf using transformer fit bag of words
from sklearn.feature_extraction.text import TfidfTransformer
tdidf_transformer = TfidfTransformer().fit(msg_bagofwords)


# In[15]:

# Check the number of rows and column in tfidf
message_tfidf = tdidf_transformer.transform(msg_bagofwords)
print message_tfidf.shape


# In[16]:

# choose naive bayes model to detect spam
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(message_tfidf, spam_data['Response'])


# In[17]:

# Check model for predicted and expected value
message = spam_data['Message'][2]
bag_of_words_for_message = bagofwords.transform([message])
tfidf = tdidf_transformer.transform(bag_of_words_for_message)

print 'predicted: ', spam_detect_model.predict(tfidf)[0]
print 'expected: ', spam_data.Response[2]


# In[ ]:



