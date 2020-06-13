# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %%
df=pd.read_csv('FuelConsumption.csv')

# %%
df.head()

# %%
df.describe().T

# %%
df.info()

# %%
n_df=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',
       'FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']]

# %%
plt.figure(figsize=(12,6))
viz = df[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()

# %%
n_df.isnull().sum()

# %%
n_df.corr()

# %%
n_df.corr()['CO2EMISSIONS'].sort_values(ascending=False)

# %%
sns.scatterplot(x='CO2EMISSIONS',y='FUELCONSUMPTION_CITY',data=n_df)

# %%
sns.scatterplot(x='CO2EMISSIONS',y='FUELCONSUMPTION_COMB_MPG',data=n_df)

# %%
sns.scatterplot(x='CO2EMISSIONS',y='CYLINDERS',data=n_df)

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# %%
X=n_df
y=df['CO2EMISSIONS']

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

# %%
lm=LinearRegression()

# %%
lm.fit(X_train,y_train)

# %%
predictions=lm.predict(X_test)

# %%
sns.set_style('whitegrid')
plt.scatter(predictions,y_test)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')

# %%
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("R2 Score : ",lm.score(X_test,y_test))

# %%
coeff=pd.DataFrame(lm.coef_,X.columns)
coeff.columns=['Coefficients']
coeff.head()

# %%
