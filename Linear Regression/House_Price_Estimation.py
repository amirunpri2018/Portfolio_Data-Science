#IMPORTING ALL NECESSARY LIBRARIES:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Importing Dataset And Basic Overview:

df=pd.read_csv('USA_Housing.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

"""
After Viewing This Data,we can analyse that there are no null values in this Dataset .There is only 1 text type feature and
rest all are numeric features.
"""


"""
Let's Do Some Exploratory Data Analysis (EDA) :
"""


sns.pairplot(data=df)

"""
Let us see some variation in Price feature to visualize it:
"""
sns.set_style('whitegrid')
sns.distplot(df['Price'])
"""
Let us see whether there is any correlation between any feature or not:
"""

df.corr()
df.corr()['Price'].sort_values(ascending=False)
"""On the basis of 'Price' Feature we can see that 'Avg.Area Income has a nearest and good positive correlation. Now 
   let us make a heat map to visualise it:
 """
sns.heatmap(df.corr(),annot=True)
sns.regplot(x='Price',y='Avg. Area Income',data=df)

"""
Now,let us define our variables for training our linear  regression model and fit our data:
"""

from sklearn.model_selection import train_test_split
from sklearn import metrics


X=df.drop('Address',axis=1)
y=df['Price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression


lm=LinearRegression()
lm.fit(X_train,y_train)


predict=lm.predict(X_test)
plt.scatter(predict,y_test)
predict.head()


print('MAE:', metrics.mean_absolute_error(y_test, predict))
print('MSE:', metrics.mean_squared_error(y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))


coef=pd.DataFrame(lm.coef_,X.columns)
coef.columns=['Coefficients']
coef.head()


