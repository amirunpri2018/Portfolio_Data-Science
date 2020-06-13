# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %%
df=pd.read_csv('titanic_train.csv')

# %%
df.head(10)

# %%
df.info()

# %%
df.describe().T

# %%
df.isnull().sum()

# %%
sns.heatmap(df.isnull(),yticklabels=False)

# %%
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df)

# %%
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df,hue='Sex')

# %%
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df,hue='Pclass')

# %%
sns.distplot(df['Age'].dropna(),kde=False)

# %%
df['Age'].hist()

# %%
sns.countplot(x='SibSp',data=df)

# %%
df['Fare'].hist(bins=40,figsize=(10,6))

# %%
"""
<b>Data Cleaning Section:</b>
"""

# %%
plt.figure(figsize=(12,6))
sns.boxplot(x='Pclass',y='Age',data=df)

# %%
"""
Let us make a function that will fill missing values of Age with average values:
"""

# %%
def fill_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        
        elif Pclass==2:
            return 29
        
        else:     
        
            return 24
    else:
        
        return Age

# %%
df['Age']=df[['Age','Pclass']].apply(fill_age,axis=1)
print("Function Successfully executed")

# %%
df.head(10)

# %%
df.isnull().sum()

# %%
sns.heatmap(df.isnull(),yticklabels=False)

# %%
df.drop('Cabin',axis=1,inplace=True)

# %%
df.head()

# %%


# %%
"""
<h3>Converting Categorical Features
We'll need to convert categorical features to dummy variables using pandas : </h3>
"""

# %%
sex=pd.get_dummies(df['Sex'],drop_first=True)
embark=pd.get_dummies(df['Embarked'],drop_first=True)

# %%
df.drop(['Sex','Embarked','Ticket','Name'],axis=1,inplace=True)

# %%
df=pd.concat([df,sex,embark],axis=1)

# %%
df.head(10)

# %%
"""
<h3>Training Our Model:</h3>
"""

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

# %%
X=df.drop('Survived',axis=1)
y=df['Survived']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# %%
lm=LogisticRegression()

# %%
lm.fit(X_train,y_train)

# %%
predict=lm.predict(X_test)

# %%
X_test


# %%
print(X_test.shape)

# %%
"""
"""
This is just a test value.Since we have 9 features , you can go ahead and enter any 9 values as you want. But be careful, try to look over decimal values of some features as well.
"""
"""

# %%
test=[[830,1,37.0,0,1,300.0000,1,0,1]]


# %%
lm.predict(test)

# %%
predict

# %%


# %%
df1=pd.DataFrame({'Actual Values':y_test,'Predicted Value':predict})
l=[]
for x in predict:
    if x==0:
        l.append("Not Survived")
    else:
        l.append("Survived")
df1['Survived OR Not Survived']=l
df1.head(10)

# %%


# %%
"""
<h3>Evaluation : </h3>
"""

# %%
print(classification_report(y_test,predict))

# %%
print(confusion_matrix(y_test,predict))

# %%
