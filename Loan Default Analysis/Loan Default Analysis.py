# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df=pd.read_csv('train.csv')

# %%
df.head(10)

# %%
"""
<h2>Understanding the various features (columns) of the dataset.</h2>
"""

# %%
df.describe().T

# %%
df.info()

# %%
df.isnull().sum()

# %%
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True)

# %%
df['Property_Area'].value_counts()

# %%
sns.set_style('whitegrid')
df['ApplicantIncome'].plot(kind='hist')
plt.xlabel('Applicant Income')

# %%
df.boxplot(column='ApplicantIncome')
#There are some outliers...


# %%
plt.figure(figsize=(16,8))
df.boxplot(column='ApplicantIncome',by='Education')
plt.ylabel('Applicant Income')

# %%
df['LoanAmount'].hist()
plt.xlabel('Loan Amount')

# %%
df.boxplot(column='LoanAmount', by = 'Gender')
plt.ylabel('Loan Amount')

# %%
"""
<h3>Understanding Categorical Variables : </h3>
"""

# %%
loan_app=df['Loan_Status'].value_counts()['Y']
loan_app
#Print total number of loan approvals

# %%
df['Self_Employed'].value_counts()

# %%
df['Self_Employed'].fillna('No',inplace=True)

# %%
df['LoanAmount'].hist()
plt.xlabel('Loan Amount')

# %%
# Impute missing values for Gender
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)

# Impute missing values for Married
df['Married'].fillna(df['Married'].mode()[0],inplace=True)

# Impute missing values for Dependents
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

# Impute missing values for Credit_History
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)


# %%
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# %%
# Convert all non-numeric values to number
cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status']

for var in cat:
    le = preprocessing.LabelEncoder()
    df[var]=le.fit_transform(df[var].astype('str'))
df.dtypes

# %%
df.head()

# %%
df.isnull().sum()

# %%
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)

# %%
df.isnull().sum()

# %%
"""
<h2> Train and Test Of Model : </h2>
"""

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

# %%
lm=LogisticRegression()

# %%
X=df.drop(['Loan_Status','Loan_ID'],axis=1)
y=df['Loan_Status']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# %%
lm.fit(X_train,y_train)

# %%
predict=lm.predict(X_test)

# %%
df1=pd.DataFrame({'Actual Value':y_test,'Predicted Value':predict})
df1.head(10)

l=[]
for x in predict:
    if x==1:
        l.append('Approved')
    else:
        l.append('Not Approved')
df1['Loan_Approval_Status']=l
df1.head(10)

# %%
"""
<h2>Metrics:</h2>
"""

# %%
print(classification_report(y_test,predict))

# %%
print(confusion_matrix(y_test,predict))

# %%
