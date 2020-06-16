# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df=pd.read_csv('Iris.csv')

# %%
df.head(10)

# %%
df.shape

# %%
df.info()

# %%
df.describe().T

# %%
df.isnull().sum()

# %%
df.drop('Id',axis=1,inplace=True)

# %%
df.head()

# %%
df.groupby('Species').size()

# %%
sns.pairplot(data=df)

# %%
df.corr()

# %%
plt.figure(figsize=(12,4))
sns.set_style('darkgrid')
sns.countplot('Species',data=df)

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# %%
X=df.iloc[:,:4]
y=df['Species']

# %%
X.columns

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60)

# %%
from sklearn.preprocessing import StandardScaler

# %%
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# %%
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

# %%
knn.predict(X_test)

# %%
pred=knn.predict(X_test)

# %%
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# %%
round(accuracy_score(y_test,pred),3)

# %%
# Finding Good Value Of K:

error_rate=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_k=knn.predict(X_test)
    error_rate.append(np.mean(pred_k!=y_test))
    
    
    

    

# %%



sns.set_style('whitegrid')
plt.figure(figsize=(12,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',
        markerfacecolor='red', markersize=10)
plt.title('Error Rate VS K Values')
plt.xlabel('K Values')
plt.ylabel('Error Rate')

# %%
print(confusion_matrix(y_test,pred))

# %%
print(classification_report(y_test,pred))

# %%
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# %%
from sklearn.metrics import accuracy_score

# %%
print('Accuracy Score : ', round(accuracy_score(y_test,pred),2))

# %%
