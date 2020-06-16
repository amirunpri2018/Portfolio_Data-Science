# %%
"""
<h2>K Nearest Neighbors with Python:</h2><br>
You've been given a classified data set from a company! They've hidden the feature column names but have given you the data and the target classes.<br>

We'll try to use KNN to create a model that directly predicts a class for a new data point based off of the features.<br>

Let's grab it and use it!
"""

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# %%
df=pd.read_csv('Classified Data',index_col=0)

# %%
df.head()

# %%
"""
<h2>Standarisisng The Variables:</h2>
"""

# %%
from sklearn.preprocessing import StandardScaler

# %%
scaler=StandardScaler()

# %%
scaler.fit(df.drop('TARGET CLASS',axis=1))

# %%
scaled_feat=scaler.transform(df.drop('TARGET CLASS',axis=1))

# %%
scaled_feat

# %%
df_feat=pd.DataFrame(scaled_feat,columns=df.columns[:-1])

# %%
df_feat.head()

# %%
"""
<h2>Train Test Split : </h2>
"""

# %%
from sklearn.model_selection import train_test_split


# %%
X=df_feat
y=df['TARGET CLASS']

# %%
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# %%
"""
<h2> Using KNN : </h2>
"""

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
knn=KNeighborsClassifier(n_neighbors=1)

# %%
knn.fit(X_train,y_train)

# %%
pred=knn.predict(X_test)

# %%
pred

# %%
"""
<h2>Evaluations : </h2>
"""

# %%
from sklearn.metrics import confusion_matrix,classification_report

# %%
print(confusion_matrix(y_test,pred))

# %%
print(classification_report(y_test,pred))

# %%
"""
<h2>Choosing a Good K Value
</h2><br>
Let's go ahead and use the elbow method to pick a good K Value:
"""

# %%
error_rate=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# %%
sns.set_style('whitegrid')
plt.figure(figsize=(12,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',
        markerfacecolor='red', markersize=10)
plt.title('Error Rate VS K Values')
plt.xlabel('K Values')
plt.ylabel('Error Rate')

# %%
knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=8')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# %%
df1=pd.DataFrame({'Actual Class':y_test,'Predicted Class':pred})
df1.head(10)

# %%


# %%
