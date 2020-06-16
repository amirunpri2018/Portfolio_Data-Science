# %%
"""
# Iris DataSet Species Detetction USING Support Vector Macine ( SVM ) And GridSearchCV :
"""

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
from sklearn.datasets import load_iris

# %%
iris=load_iris()

# %%
iris.keys()

# %%
print(iris['DESCR'])

# %%
iris['data']

# %%
iris['target']

# %%
iris['target_names']

# %%
iris['feature_names']

# %%
df=pd.DataFrame(iris['data'],columns=iris['feature_names'])
df.head()

# %%
df['Target Class']=iris['target']


# %%
l=[]
for x in df['Target Class']:
    if x==0:
        l.append('setosa')
    elif x==1:
        l.append('versicolor')
    else:
        l.append('virginica')
df['species']=l

# %%
df.head()

# %%
df.info()

# %%
df.describe().T

# %%
df.isnull().sum() #To Check For Null Values

# %%
"""
## Exploratory Data Analysis :
"""

# %%
sns.set_style('darkgrid')
sns.pairplot(df,hue='species')

# %%
setosa = df[df['species']=='setosa']
sns.kdeplot(setosa['sepal width (cm)'],setosa['sepal length (cm)'],cmap='plasma',shade=True,shade_lowest=False)
plt.title('Setosa Species KDE Plot')

# %%
versicolor = df[df['species']=='versicolor']
sns.kdeplot(versicolor['sepal width (cm)'],versicolor['sepal length (cm)'],cmap='plasma',shade=True,shade_lowest=False)
plt.title('Versicolor Species KDE Plot')

# %%

virginica = df[df['species']=='virginica']
sns.kdeplot(virginica['sepal width (cm)'],virginica['sepal length (cm)'],cmap='plasma',shade=True,shade_lowest=False)
plt.title('Virginica Species KDE Plot')

# %%
df.corr()['Target Class'].sort_values(ascending=False)

# %%
sns.heatmap(df.corr(),annot=True)

# %%
"""
## Train Test Split:
"""

# %%
from sklearn.model_selection import train_test_split

# %%
X=df.drop(['Target Class','species'],axis=1)
y=df['species']

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

# %%
"""
## SVC Model Train :
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
"""
## Model Evaluation :
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
## Using GridSearchCV For Better Parameters :
"""

# %%
from sklearn.model_selection import GridSearchCV

# %%
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.001,0.0001]}

# %%
grid=GridSearchCV(SVC(),param_grid,verbose=3,refit=True)

# %%
grid.fit(X_train,y_train)

# %%
grid.best_estimator_

# %%
grid.best_params_

# %%
grid_pred=grid.predict(X_test)

# %%
df1=pd.DataFrame({'Actual Species':y_test,'Predicted Species':grid_pred})
df1.head(10)

# %%
"""
### TEST CASE ( ANY RANDOM VALUE TESTING EXAMPLE ) :
"""

# %%
t_case=[[6.2,3.8,1.6,0.4]] #Any 4 Random Features as per our dataframe
v=grid.predict(t_case)
print(v)

# %%
"""
## Evaluation for GridSearchCV Parameters :
"""

# %%
print('The Classification Report is :')
print('\n')
print(classification_report(y_test,grid_pred))
print('\n')
print("Confusion Matrix : ")
print('\n')
print(confusion_matrix(y_test,grid_pred))
print('\n')
print('The Accuracy Is : ',round(accuracy_score(y_test,grid_pred),2))

# %%
"""
#### As You Can See, by Performing Grid Search CV , we found suitable parameters for refit of our model and our accuracy has been increased to 0.98...therefore..increasing our model performeance ...!!!!
"""

# %%
