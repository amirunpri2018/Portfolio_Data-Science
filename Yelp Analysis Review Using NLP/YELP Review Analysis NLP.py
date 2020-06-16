# %%
"""
# YELP Review Analysis Project :  
"""

# %%
"""
### In this NLP project we will be attempting to classify Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews.
"""

# %%
"""
#### Each observation in this dataset is a review of a particular business by a particular user.

The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.

The "cool" column is the number of "cool" votes this review received from other Yelp users.

All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.

The "useful" and "funny" columns are similar to the "cool" column.
"""

# %%
"""
RecSys2013: Yelp Business Rating Prediction :
https://www.kaggle.com/c/yelp-recsys-2013
    
    Dataset Link By Kaggle
"""

# %%
"""
### Importing Libraries :
"""

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df=pd.read_csv('yelp.csv')

# %%
df.head()

# %%
df.describe().T

# %%
df.isnull().sum()

# %%
"""
## Basic Data Exploratory Analysis :
"""

# %%
df['text-length']=df['text'].apply(len)

# %%
df['text-length'].head()

# %%
sns.set_style('darkgrid')
g=sns.FacetGrid(df,col='stars')
g.map(plt.hist,'text-length')

# %%
sns.boxplot(x='stars',y='text-length',data=df)

# %%
sns.countplot(x='stars',data=df)

# %%
stars=df.groupby('stars').mean()
stars

# %%
df.corr()

# %%
sns.heatmap(df.corr(),annot=True)

# %%
df.corr()['stars'].sort_values(ascending=False)

# %%
"""
## NLP Classification :
"""

# %%
"""
##### Let us Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.
"""

# %%
yelp_c=df[(df['stars']==1) | (df['stars']==5)]

# %%
yelp_c.hist()
plt.tight_layout()

# %%
df[df['stars']==1].count()

# %%
df[df['stars']==5].count()

# %%
X=yelp_c['text']
y=yelp_c['stars']


# %%
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# %%
cv=CountVectorizer()
X=cv.fit_transform(X)

# %%
"""
## Train Test Split :
"""

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# %%
mb=MultinomialNB()

# %%
mb.fit(X_train,y_train)

# %%
"""
## Predictions and Evaluations
"""

# %%
mb.predict(X_test)

# %%
pred=mb.predict(X_test)

# %%


# %%
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# %%
print(classification_report(y_test,pred))

# %%
print('The Acuuracy Score is : ',accuracy_score(y_test,pred))

# %%
"""
##### Now we will be trying two different things to Evaluate our project:
    1. Including Tfidf
    2.Pipeline Method
"""

# %%
from sklearn.pipeline import Pipeline

# %%
pipeline=Pipeline([
    ('bow',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
    
])

# %%
"""
### Now we need to fit data in original form, since we have included our steps in Pipeline From Starting :
"""

# %%
X1=yelp_c['text']
y1=yelp_c['stars']

# %%
 X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0)

# %%
pipeline.fit(X1_train,y1_train)

# %%
"""
## Predictions And Evaluations :
"""

# %%
pipeline.predict(X1_test)

# %%
pred_p=pipeline.predict(X1_test)

# %%
print(classification_report(y1_test,pred_p))

# %%
print(accuracy_score(y1_test,pred_p))

# %%
"""
### Looks like Tfidf Transformer made our model worse..Let us remove it from our pipeline and try it again with new pipeline model :
"""

# %%
X2=yelp_c['text']
y2=yelp_c['stars']

# %%
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3,random_state=0)

# %%
pipe2=Pipeline([
    ('bow',CountVectorizer()),
    ('classifier',MultinomialNB())
    
    
])

# %%
pipe2.fit(X2_train,y2_train)

# %%
pipe2.predict(X2_test)

# %%
pred_p2=pipe2.predict(X2_test)

# %%
df7=pd.DataFrame({'Actual Values ':y2_test,'Predicted Values ':pred_p2})
df7.head(10)

# %%
"""
## Evaluations of Pipeline 2 :
"""

# %%
print(classification_report(y2_test,pred_p2))

# %%
print('The Accuracy Score :',round(accuracy_score(y2_test,pred_p2),2))

# %%
"""
### So, To summarise , our second pipeline with only countvectoriser and naive bayes algorithm worked perfectly with 91 % accuracy 
"""

# %%
