
# coding: utf-8

# In[1]:

# Import Libaries
import string
from nltk.corpus import stopwords
import pandas as pd
import re
from collections import Counter


# In[2]:

# View the stopwords
stopwords.words('english')[0:10]


# In[3]:

# Load the book "Around the world in 80 days"
ReadBook = open('C:\Users\pravinw\Documents\Fractal\DATA SCIENTIST\PractiseAssignment_Simplilearn\Around the world in 80 days.txt',"r")


# In[4]:

# Read the book line wise and remove the line breaks
Book_Lines = ReadBook.read().replace('\n',' ')


# In[5]:

# Now we remove the punctuations
no_punctuation = [char for char in Book_Lines if char not in string.punctuation]
no_punctuation


# In[6]:

# Now we compile the book back without punctuations and line breaks
no_punctuation = ''.join(no_punctuation)
no_punctuation


# In[7]:

# Here each word is splitted
no_punctuation.split()


# In[8]:

# Now lets remove the stop words
rectify_sentence = [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]
rectify_sentence


# In[9]:

# Counts each word number of times it has featured in the book
counts = Counter(rectify_sentence)
print(counts)

