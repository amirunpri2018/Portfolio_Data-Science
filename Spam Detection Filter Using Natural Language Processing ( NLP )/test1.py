import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

""" 
We'll be using a dataset from the UCI datasets!

The file we are using contains a collection of more than 5 thousand SMS phone messages.
"""

from nltk.tokenize import sent_tokenize, word_tokenize

messages = [line.rstrip() for line in open('smsspamcollection\SMSSpamCollection')]
print('Total Messages Are :', len(messages))

for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')

""" 
ham- ham means regular Message

spam- span means unwanted, commercial message



Due to the spacing we can tell that this is a TSV ("tab separated values") file, where the first column is a label saying whether the given message is a normal message (commonly known as "ham") or "spam". The second column is the message itself. (Note our numbers aren't part of the file, they are just from the enumerate call).

Using these labeled ham and spam examples, we'll train a machine learning model to learn to discriminate between ham/spam automatically. Then, with a trained model, we'll be able to classify arbitrary unlabeled messages as ham or spam

"""

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['labels', 'message'])
print(messages.head())

print('\n')

print(messages.describe())
print('\n')
print(messages.info())
print('\n')
print(messages.isnull().sum())

print('\n')

print(messages.groupby('labels').describe())

messages['length'] = messages['message'].apply(len)
print('\n')
print(messages.head())

sns.set_style('darkgrid')
messages['length'].plot(kind='hist', bins=50)
plt.xlabel(' Length Of Messages ')
plt.show()

""" 
the x-axis goes all the way to 1000ish, this must mean that there is some really long messages..
"""

print('\n')

print(messages.length.describe())

print(messages[messages['length'] == 910])

messages.hist(column='length', by='labels', bins=50)
plt.show()

""" 
Text Pre-processing


"""

""" 
First we will create a function which will remove punctuation marks from strings

and will join them to create a string without punctuation marks..
"""

import string

mess = 'Sample Message! Notice: It has punctuation'
print(mess)
print('\n')
nopunc=[char for char in mess if char not in string.punctuation]

print(nopunc)
"""
we will have a output like this:
['S', 'a', 'm', 'p', 'l', 'e', ' ', 'M', 'e', 's', 's', 'a', 'g', 'e', 'N', 'o', 't', 'i', 'c', 'e', ' ', 'I', 't', ' ', 'h', 'a', 's', ' ', 'p', 'u', 'n', 'c', 't', 'u', 'a', 't', 'i', 'o', 'n']

Let us join all this words to make a string again:
"""
nopunc=''.join(nopunc)
print('\n')
print(nopunc)

from nltk.corpus import stopwords
print('Some Top Stopwords From nltk.corpus Are :',stopwords.words('english')[0:10])
print('\n')
sep_mess=nopunc.split()
print(sep_mess)

print('\n')

clean_mess=[word for word in sep_mess if word.lower() not in stopwords.words('english')]
print(clean_mess)
print('\n')
""" 
Now let us repeat this process again on our actual dataframe:




 
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
 
    # Check characters to see if they are in punctuation


    # Join the characters again to form the string.

    
    # Now just remove any stopwords
    

"""

def text_process(mess):
    nopunc=[char for char in mess if char not in string.punctuation]

    nopunc=''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

print('Original Dataframe :')
print('\n')
print(messages.head(5))
print('\n')
print('Modified Clean Messages After Removing Stopwords : ')
print('\n')
mod=messages['message'].head(5).apply(text_process)

print(mod)
print('\n')
""" 
Vectorisation Of Words :

"""

from sklearn.feature_extraction.text import CountVectorizer


bow_transformer=CountVectorizer(analyzer=text_process).fit(messages['message'])

#Print length of voabulary of words in our Sparse Matrix formed using Count Vectoriser()
print(len(bow_transformer.vocabulary_))

#Example :
print('\n')
message5=messages['message'][4]
print(message5)
print('\n')
#Now let's see its vector representation:

bow5=bow_transformer.transform([message5])
print(bow5)
print(bow5.shape)

""" 
Now we can use .transform on our Bag-of-Words (bow) transformed object and transform the 
entire DataFrame of messages. 
Let's go ahead and check out how the bag-of-words counts for the entire SMS corpus 
is a large, sparse matrix:

"""

messages_bow=bow_transformer.transform(messages['message'])

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


from sklearn.feature_extraction.text import TfidfTransformer

tfidf=TfidfTransformer().fit(messages_bow)

messages_tfidf=tfidf.transform(messages_bow)
print(messages_tfidf.shape)

""" 
Training Model :

The Naive Bayes classifier algorithm is a good choice : 

"""


from sklearn.naive_bayes import  MultinomialNB
spam_detect_model=MultinomialNB().fit(messages_tfidf,messages['labels'])

""" 
Evaluation Of Model :
"""
tfidf5=tfidf.transform(bow5)
print('Predicted : ',spam_detect_model.predict(tfidf5[0]))
print('Actual : ',messages['labels'][4])


""" 
Model Evaluation : 

"""
all_pred=spam_detect_model.predict(messages_tfidf)
print(all_pred)
from sklearn.metrics import classification_report

print(classification_report(messages['labels'],all_pred))



"""
Now let us Create A perfect Model :


"""

from sklearn.model_selection import train_test_split

msg_train,msg_test,label_train,label_test=train_test_split(messages['message'],messages['labels'],test_size=0.2)


""" 
Creating a Data Pipeline
Let's run our model again and then predict off the test set. 
We will use SciKit Learn's pipeline capabilities to store a pipeline of workflow. 
This will allow us to set up all the transformations that we will do to the data for future use.

"""

from sklearn.pipeline import Pipeline

pipeline=Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB()),
])

pipeline.fit(msg_train,label_train)


predictions=pipeline.predict(msg_test)

print(classification_report(predictions,label_test))


