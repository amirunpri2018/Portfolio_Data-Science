"""Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing
advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go
home and order either on a mobile app or website for the clothes they want.

The company is trying to decide whether to focus their efforts on their mobile app experience or their website.
They've hired you on contract to help them figure it out! """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('Ecommerce Customers')
print(df.head())

print(df.info())
print(df.isnull().sum())
print(df.describe().T)

sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df)
plt.show()

sns.set_style('whitegrid')
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)
plt.show()

sns.distplot(df['Time on Website'],kde=False)
plt.show()
sns.distplot(df['Time on App'],kde=False)
plt.show()

sns.pairplot(df)
plt.show()

sns.regplot(x='Length of Membership',y='Yearly Amount Spent',data=df)
plt.show()

print(df.corr())

print(df.corr()['Yearly Amount Spent'].sort_values(ascending=False))

sns.heatmap(df.corr(),annot=True)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X=df.drop(['Email','Address','Avatar','Yearly Amount Spent'],axis=1)
y=df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm=LinearRegression()
lm.fit(X_train,y_train)

predict=lm.predict(X_test)

print(predict)

plt.scatter(predict,y_test)
plt.xlabel("Predictions")
plt.ylabel("Actual Values")
plt.show()



print('MAE:', metrics.mean_absolute_error(y_test, predict))
print('MSE:', metrics.mean_squared_error(y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))
print("R2 Score : ",lm.score(X_test,y_test))

coeff=pd.DataFrame(lm.coef_,X.columns,columns=['Coefficients'])
print(coeff)


"""
** How can you interpret these coefficients? **

Interpreting the coefficients:

Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.
Do you think the company should focus more on their mobile app or on their website?

This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the 
mobile app, or develop the app more since that is what is working better. This sort of answer really depends on the 
other factors going on at the company, you would probably want to explore the relationship between Length of 
Membership and the App or the Website before coming to a conclusion! 

"""