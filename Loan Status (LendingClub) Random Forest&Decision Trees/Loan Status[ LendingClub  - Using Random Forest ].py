# %%
"""
# Random Forest Project -> LendingLoan Club :
"""

# %%
"""
For this project we will be exploring publicly available data from LendingClub.com. Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
"""

# %%
"""
We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full.<br>

Here are what the columns represent:<br>

credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.<br>
purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").<br>
int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.<br>
installment: The monthly installments owed by the borrower if the loan is funded.<br>
log.annual.inc: The natural log of the self-reported annual income of the borrower.<br>
dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).<br>
fico: The FICO credit score of the borrower.<br>
days.with.cr.line: The number of days the borrower has had a credit line.<br>
revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).<br>
revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).<br>
inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.<br>
delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.<br>
pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).
"""

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df=pd.read_csv('loan_data.csv')

# %%
df.head()

# %%
df.info()

# %%
df.describe().T

# %%
df.isnull().sum()

# %%


# %%
"""
## Exploratory Data Analysis : 
"""

# %%
df.corr()['not.fully.paid'].sort_values(ascending=False)

# %%
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True)

# %%
df[df['credit.policy']==1]['fico'].hist(label='Credit-Policy=1')
df[df['credit.policy']==0]['fico'].hist(label='Credit-Policy=0')
plt.legend()
plt.xlabel("FICO")

# %%
df[df['not.fully.paid']==1]['fico'].hist(label='Credit-Policy=1',bins=30)
df[df['not.fully.paid']==0]['fico'].hist(label='Credit-Policy=0',bins=30)
plt.legend()
plt.xlabel('FICO')

# %%
plt.figure(figsize=(12,6))
sns.set_style('whitegrid')
sns.countplot(x='purpose',hue='not.fully.paid',data=df)
plt.legend()

# %%
sns.jointplot(x='fico',y='int.rate',data=df)

# %%
plt.figure(figsize=(12,4))
sns.lmplot(x='fico',y='int.rate',data=df,hue='credit.policy',col='not.fully.paid')

# %%
"""
## Check For Data : 
"""

# %%
df.info()

# %%
df.isnull().sum()    #Checking For Null Values:

# %%
"""
## Dealing With Categorical Features : 
"""

# %%
df.groupby('purpose').sum().T

# %%
cat_feat=['purpose']

# %%
final_df=pd.get_dummies(df,columns=cat_feat,drop_first=True)

# %%
final_df.info()

# %%
final_df.head(10)

# %%


# %%
"""
## Train Test Split : 
"""

# %%
from sklearn.model_selection import train_test_split

# %%
X=final_df.drop('not.fully.paid',axis=1)
y=final_df['not.fully.paid']

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

# %%
"""
## Decision Tree Model : 
"""

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
dtree=DecisionTreeClassifier()

# %%
dtree.fit(X_train,y_train)

# %%
dtree.predict(X_test)

# %%
predict=dtree.predict(X_test)

# %%
df1=pd.DataFrame({'Actual Class':y_test,'Predicted Class':predict})
df1.head()

# %%
l=[]
for x in df1['Predicted Class']:
    if x==1:
        l.append('Not Fully Paid')
    else:
        l.append('Fully Paid')
df1['Loan Status Class']=l

# %%
df1.head()

# %%
plt.figure(figsize=(12,6))
t=df1.head(25,)
t.hist()


# %%
"""
## TEST CASE 1: 
"""

# %%
#The details mentioned below is just a random test case with random values:
tc_1=pd.DataFrame([[0,0.1201,980,14.252631,20.00,805,6490.000000,39000,25.2,0,0,0,0,0,0,0,1,0]])
tc_1.shape

# %%
vl_r=dtree.predict(tc_1)
print(vl_r)
for x in vl_r:
    if x==0:
        print('NOT FULLY PAID')
    else:
        print('PAID')

# %%
"""
## Metrics Performance Evaluation ( Decision Tree ) :
"""

# %%
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# %%
print('Confusion Matrix :')
print('\n')
print(confusion_matrix(y_test,predict))
print('\n')
print('Overall Classification Report : ')
print('\n')
print(classification_report(y_test,predict))
print('\n')
print('The Accuracy Score : ',round(accuracy_score(y_test,predict),2))

# %%
"""
## Random Forest Model :
"""

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
rfc=RandomForestClassifier(n_estimators=200) #you can choose and play with any value for n_estimators

# %%
rfc.fit(X_train,y_train)

# %%
pred_1=rfc.predict(X_test)

# %%
df2=pd.DataFrame({'Actual Class':y_test,'Predicted Class':pred_1})
df2.head(10)

# %%
l=[]
for x in df2['Predicted Class']:
    if x==1:
        l.append('Not Fully Paid')
    else:
        l.append('Fully Paid')
df2['Loan Status Class']=l

# %%
df2.head()

# %%
plt.figure(figsize=(12,6))
t=df2
t.hist()


# %%
"""
## Test Case 2 :
"""

# %%
df2.shape

# %%
X.head(2)

# %%
X.shape

# %%
#The details mentioned below is just a random test case with random values:
tc_2=pd.DataFrame([[1,0.1001,900,12.252631,16.00,745,5000.000000,45000,45.2,0,0,0,0,1,0,0,0,0]])
tc_2.shape

# %%
"""
Since the feature shape columns match...(18)...we can perform predictions and we can see results :
"""

# %%
# This will print the status whther the loan is fully paid or not for the above mentioned test case:
vl_p=rfc.predict(tc_1)
print(vl_p)
for x in vl_p:
    if x==0:
        print('NOT FULLY PAID')
    else:
        print('PAID')


# %%


# %%
"""
## Metrics Performance Evaluation ( Random Forest ) :
"""

# %%
print('Confusion Matrix :')
print('\n')
print(confusion_matrix(y_test,pred_1))
print('\n')
print('Overall Classification Report : ')
print('\n')
print(classification_report(y_test,pred_1))
print('\n')
print('The Accuracy Score : ',round(accuracy_score(y_test,pred_1),2))

# %%
"""
Hence, Our Random Forest Has much better Accuracy Score , so , Random Forest performed Much Better..!!
"""

# %%
