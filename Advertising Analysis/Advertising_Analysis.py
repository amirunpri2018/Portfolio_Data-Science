# %%
"""
<h1>Advertising Analysis</h1>
"""

# %%
"""
In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.

This data set contains the following features:

'Daily Time Spent on Site': consumer time on site in minutes<br>
'Age': cutomer age in years<br>
'Area Income': Avg. Income of geographical area of consumer<br>
'Daily Internet Usage': Avg. minutes a day consumer is on the internet<br>
'Ad Topic Line': Headline of the advertisement<br>
'City': City of consumer<br>
'Male': Whether or not consumer was male<br>
'Country': Country of consumer<br>
'Timestamp': Time at which consumer clicked on Ad or closed window<br>
'Clicked on Ad': 0 or 1 indicated clicking on Ad<br>
"""

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df=pd.read_csv('advertising.csv')

# %%
df.head()

# %%
df.info()

# %%
df.isnull().sum()

# %%
df.describe().T

# %%

df.groupby('Country')['Clicked on Ad'].count().sort_values(ascending=False)

# %%
sns.set_style('whitegrid')
df['Age'].plot(kind='hist')
plt.xlabel('Age')

# %%
sns.jointplot(x='Age',y='Area Income',data=df)

# %%
sns.set_style('whitegrid')
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=df,kind='kde')

# %%
sns.set_style('whitegrid')
sns.jointplot(y='Daily Internet Usage',x='Daily Time Spent on Site',data=df)

# %%
sns.pairplot(data=df,hue='Clicked on Ad')

# %%
"""
<h3>Logistic Regression</h3>
<br>
Now it's time to do a train test split, and train our model!
"""

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

# %%
X=df.drop(['Ad Topic Line','City','Timestamp','Country','Clicked on Ad'],axis=1)
y=df['Clicked on Ad']

# %%
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# %%
log=LogisticRegression()

# %%
log.fit(X_train,y_train)

# %%
"""
<h2>Predictions : </h2>
"""

# %%
predict=log.predict(X_test)

# %%
predict

# %%
df1=pd.DataFrame({'Actual Value':y_test,'Predicted Value':predict})
df1.head()

# %%
vcat=[]
for x in predict:
    if x==1:
        vcat.append('Yes,Clicked!')
    else:
        vcat.append('Not Clicked!')

df1['Clicked Or Not']=vcat
df1.head(10)

# %%
X_test.columns

# %%
X_test.head(2)


# %%
X_test.shape

# %%
"""
Now Let us supoose we have a person , SAY, Mr.Sam, whose features are as follows:<br>
    Daily Time spent on site :30.00 hrs<br>
    Age :45<br>
    Area Income :60000.20<br>
    Daily Internet Usage :300.00<br>
    Male :1(Yes)<br>
        
    Let us predict whether he would have clicked on Ad or not , based on our Prediction Model:
"""

# %%
testcase=[[30.00,45,60000.20,300.00,1]]

# %%
tp=log.predict(testcase)
tp

# %%
"""
<h2>Evaluation Metrics: </h2>
"""

# %%
print(classification_report(y_test,predict))

# %%
print(confusion_matrix(y_test,predict))

# %%


# %%
