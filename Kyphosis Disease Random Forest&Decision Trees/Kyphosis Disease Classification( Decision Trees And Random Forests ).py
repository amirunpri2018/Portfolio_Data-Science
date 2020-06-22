# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df=pd.read_csv('kyphosis.csv')

# %%
df.head()

# %%
df.info()

# %%
df.describe().T

# %%
df.isnull().sum()

# %%
sns.pairplot(df,hue='Kyphosis')

# %%
sns.heatmap(df.corr(),annot=True)

# %%
sns.set_style('whitegrid')
sns.barplot(y='Age',x='Number',data=df)

# %%
from sklearn.model_selection import train_test_split

# %%
X=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']

# %%
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %%
"""
## Decision Tree :
"""

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
dtree=DecisionTreeClassifier()

# %%
dtree.fit(X_train,y_train)

# %%
predict=dtree.predict(X_test)

# %%
predict

# %%
df_n=pd.DataFrame({'Actual Value:':y_test,'Predicted Value':predict})
df_n.head()

# %%


# %%
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# %%
print(confusion_matrix(y_test,predict))
print('\n')
print(classification_report(y_test,predict))
print('\n')
print('The Accuracy is : ',accuracy_score(y_test,predict))

# %%
"""
## Random Forest :
"""

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rfc=RandomForestClassifier(n_estimators=100)

# %%
rfc.fit(X_train,y_train)

# %%
predict_1=rfc.predict(X_test)

# %%
print(confusion_matrix(y_test,predict_1))
print('\n')
print(classification_report(y_test,predict_1))
print('\n')
print('The Accuracy is : ',accuracy_score(y_test,predict_1))

# %%
