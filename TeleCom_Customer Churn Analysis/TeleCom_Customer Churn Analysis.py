# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn import preprocessing

# %%
df=pd.read_csv('ChurnData.csv')

# %%
df.head()

# %%
df['churn']=df['churn'].astype(int)

# %%
df.head()

# %%
df.info()

# %%
df.describe().T

# %%
df.corr()['churn'].sort_values(ascending=False)

# %%
plt.figure(figsize=(18,10))
sns.heatmap(df.corr(),annot=True)

# %%
sns.set_style('whitegrid')
df['age'].plot(kind='hist')
plt.xlabel('Age')

# %%
plt.figure(figsize=(10,5))
sns.barplot(y='age',x='churn',data=df)

# %%
sns.barplot(y='age',x='internet',data=df)

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

# %%
X=df.drop('churn',axis=1)
y=df['churn']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
log=LogisticRegression()

# %%
log.fit(X_train,y_train)

# %%
predict=log.predict(X_test)

# %%
predict

# %%
df1=pd.DataFrame({'Actual Values':y_test,'Predicted Values':predict})
df1.head()

# %%
vcat=[]
for x in predict:
    if x==1:
        vcat.append('Yes')
    else:
        vcat.append('No')

df1['Churn Status']=vcat
df1.head(10)

# %%
"""
<h2>Evaluation Metrics: </h2>
"""

# %%
print(classification_report(y_test,predict))

# %%
print(confusion_matrix(y_test,predict))

# %%
