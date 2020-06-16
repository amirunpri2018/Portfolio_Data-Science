# %%
"""
# Cancer Detetcion- Maligant / Benign USING Support Vector Machines Classifier ( SVM ) :
"""

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
from sklearn.datasets import load_breast_cancer

# %%
cancer=load_breast_cancer()

# %%
cancer.keys()

# %%
print(cancer['DESCR'])

# %%
cancer['data']

# %%
cancer['target']

# %%
cancer['target_names']

# %%
cancer['feature_names']

# %%
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

# %%
df.head()

# %%
df['Cancer']=pd.DataFrame(cancer['target'],columns=['Cancer'])

# %%
df.head()

# %%
df.info()

# %%
df.describe().T

# %%
df.isnull().sum()

# %%
"""
## Exploratory Data Analysis :
"""

# %%
df.corr()['Cancer'].sort_values(ascending=False)

# %%
plt.figure(figsize=(18,9))
sns.heatmap(df.corr(),annot=True)

# %%
sns.set_style('whitegrid')
sns.barplot(x='Cancer',y='smoothness error',data=df)

# %%
sns.barplot(y='mean fractal dimension',x='Cancer',data=df)

# %%
sns.jointplot(x='mean fractal dimension',y='smoothness error',data=df)

# %%
"""
## Train Test Split :
"""

# %%
from sklearn.model_selection import train_test_split

# %%
X=df.drop('Cancer',axis=1)
y=df['Cancer']

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

# %%
X.columns

# %%
X.shape

# %%
y.shape

# %%
y

# %%
"""
## Support Vector Machine ( SVM ):
"""

# %%
from sklearn.svm import SVC

# %%
model=SVC()

# %%
model.fit(X_train,y_train)

# %%
model.predict(X_test)

# %%
predict=model.predict(X_test)

# %%
df_c=pd.DataFrame({'Actual Class':y_test,'Predicted Class':predict})
df_c.head()

# %%
l=[]

for x in df_c['Predicted Class']:
    if x==1:
        l.append('Maligant')
    else:
        l.append('Benign')
df_c['Maligant/Benign']=l

# %%
df_c.head()

# %%
"""
## Metrics Evaluation :
"""

# %%
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# %%
print('The Classification Report is :')
print('\n')
print(classification_report(y_test,predict))
print('\n')
print("Confusion Matrix : ")
print('\n')
print(confusion_matrix(y_test,predict))
print('\n')
print('The Accuracy Is : ',round(accuracy_score(y_test,predict),2))

# %%
"""
## GridSearchCV - Finding Better Parameters :
"""

# %%
from sklearn.model_selection import GridSearchCV

# %%
param_grid={'C':[0.1,1,10,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

# %%
grid=GridSearchCV(SVC(),param_grid,verbose=3)

# %%
grid.fit(X_train,y_train)

# %%
grid.best_params_

# %%
grid.best_estimator_

# %%
grid_predict=grid.predict(X_test)

# %%
print('The Classification Report is :')
print('\n')
print(classification_report(y_test,grid_predict))
print('\n')
print("Confusion Matrix : ")
print('\n')
print(confusion_matrix(y_test,grid_predict))
print('\n')
print('The Accuracy Is : ',round(accuracy_score(y_test,grid_predict),2))

# %%
"""
Since the Accuracy we got from Grid Search CV is much better .. 0.94 , therefore, we can use GridsearchCV for  finding the best parameters
"""

# %%
