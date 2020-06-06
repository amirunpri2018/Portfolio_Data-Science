"""
This is boston dataset which has been taken from kaggle.com
Here we will try to predict house prices based on features from the dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
boston=load_boston()
print(boston.DESCR)
print(boston.keys())

boston_df=pd.DataFrame(boston.data)
boston_df.columns=boston.feature_names
print(boston_df.head())
boston_df['Price']=pd.DataFrame(boston.target)

print(boston_df.head(10))

print(boston_df.describe())
print(boston_df.info())
print(boston_df.isnull().sum())


plt.figure(figsize=(12,4))
sns.set_style('whitegrid')
plt.hist(boston_df['Price'],bins=30)
plt.xlabel('House Prices in $1000')
plt.show()

print(boston_df.corr())
print(boston_df.corr()['Price'].sort_values(ascending=False))

sns.scatterplot(x='RM',y='Price',data=boston_df)
plt.show()

sns.scatterplot(x='LSTAT',y='Price',data=boston_df)
plt.show()

plt.figure(figsize=(12,4))
sns.heatmap(boston_df.corr(),annot=True)
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X=boston_df.drop('Price',axis=1)
y=boston_df['Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

lm=LinearRegression()
lm.fit(X_train,y_train)
predict=lm.predict(X_test)
print(predict)

plt.figure(figsize=(10,5))
plt.scatter(predict,y_test)
plt.xlabel('Predicted Prices')
plt.ylabel('Actual Prices')
plt.show()

print("R2 Score : ",lm.score(X_test,y_test))
print("Mean Absolute Error : ",metrics.mean_absolute_error(y_test,predict))
print("Mean Squared Error : ",metrics.mean_squared_error(y_test,predict))
print("Root Mean Squared Error : ",np.sqrt(metrics.mean_squared_error(y_test,predict)))



coeff=pd.DataFrame(lm.coef_,X.columns)
coeff.columns=['Coefficients']
print(coeff.head())
