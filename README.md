# customer_churn_prediction
# A) Data Manupulation

# B) Data Visualization 

# C)Machine Learning with Nrural Network

# C1) Data Prediction using Linear Regression  

# C2) Data Prediction using Logistic Regression

# C3)Data Prediction using Multiple-Logistic Regression

# C4)Data Prediction using Decision Tree

# C5) Data Prediction using Random Forest 



## Code Examples



#Import required Libraries for data manupulation 
import pandas as pd

#A1)Read CSV file using Pandas and strore it as "data"
data=pd.read_csv("Customer_Churn.csv")

#A2) Visualize first 5 entries of data 
data.head()

#A3) Visualize first 20 entries of data 
data.head(20)

#A4) Extract 5th Column form data 
customer_c5=data.iloc[:,4]
customer_c5

#A5) Extract 10th row form data 
customer_r10=data.iloc[10,:]
customer_r10

#A6) Extract data for only male customer
data.head() #check data and find header as gender 
male_customers=data[data['gender']=='Male']
male_customers

#A7) Extract data for only female customer
female_customers=data[data['gender']=='Female']
female_customers

#A8)Extract the only senior citizen male customers' data whose payment method is Electronic check
senior_male_electroncs=data[(data['gender']=='Male') & (data['SeniorCitizen']==1) & (data['PaymentMethod']=='Electronic check')]
senior_male_electroncs

#A9) Extract the data of customers who have completed 70 years or have paid Monthly charges more than 100$
tenure_data=data[(data['tenure']>70)|(data['MonthlyCharges']>100)]
tenure_data

#A10) Extract the data of customers whose contarct is two years, payment method is Mailed check and received churn return 
two_mail_yes=data[(data['Contract']=='Two year') & (data['PaymentMethod']=='Mailed check') & (data['Churn']=='Yes')]
two_mail_yes

#A11) Exract any 333 random customers data 
customer_333=data.sample(n=333)
#customer_333

#A12) Get the count of dirrent levels from the gender data. It will give you the keys(male,female) and the count of eack key
data['gender'].value_counts()

#A13) Get count of internet services provided to customers
data['InternetService'].value_counts()

#import required libraries for data visualization
import matplotlib.pyplot as plt

#B1) Build bar plot for count of internet serives with respect to their categories 
x=data['InternetService'].value_counts().keys()
#x
y=data['InternetService'].value_counts().tolist()
#y
plt.xlabel("Categories of internet services")
plt.ylabel("Count of categories")
plt.title("Distribution of Internet Services")
plt.bar(x,y,color="red")
plt.show()

#B2.1) Build a histogram for the tenure column
plt.hist(data['tenure'],color='orange')
plt.title("Histogram with default bins")
plt.show()
#B2.2) Specify the bins of histrogram to 75 to see data for each tenure
plt.hist(data['tenure'],bins=75)
plt.title("Histogram with 75 bins")
plt.show()

#B3) Build scatter plot between monthly chargeer and tenure (first 100 values)
plt.scatter(x=data['tenure'].head(100),y=data['MonthlyCharges'].head(100))
plt.xlabel("Tenure of Customers")
plt.ylabel("Monthly Chrages of Customers")
plt.title("Scatter Plot")
plt.show()


#B3) Build between monthly chargeer and tenure and spesify the data of male and female with diffrenet colours(first 100 values)
import seaborn as sns
sns.scatterplot(x=data['tenure'].head(100),y=data['MonthlyCharges'].head(100),hue=data['gender'].head(100))


#B4) Build box plot  between tenure and contract
data.boxplot(column='tenure', by=['Contract'])
plt.show()

#B5) Use seaborn library to build box plot  between tenure and contract

sns.boxplot(x=data['Contract'],y=data['tenure'])
plt.show()

#1)Bild a simple linear model where dependent variable is "Monthly charges" and independent variable is "tenure"

from sklearn.linear_model import LinearRegression
x=pd.DataFrame(data['tenure']) #independent variable
y=data['MonthlyCharges'] #dependent variable

#2)Devide the dataset into train and test data sets in 70:30 ratio
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.7,random_state=0) #Option 1: to select 70% data as train data
#xtrain,xtest,ytrain,ytesr=train_test_split(x,y,test_size=0.3,random_state=0) #Option 2: to select 30% data as test data
print("size of original data=",len(x))
print("size of train data=",len(xtrain))
print("size of test data=",len(xtest))

#3) Build model on training datset and test it with testing data set
#Create object (lr) with imported linear regression model
lr=LinearRegression() 
#train the model using training dataset xtrain and ytrain
lr.fit(xtrain,ytrain)

#4) Carry out the predicting/testing using trained model by passing the test data
ypred=lr.predict(xtest)
print("predicted values of y are:\n",ypred)
print("\n")
print("Actual Values of y are:\n",ytest.values)

#5) Find the error in prediction and store the eroor
from sklearn.metrics import mean_squared_error
ms_error=mean_squared_error(ytest,ypred)
print("mean square error=",ms_error)

#6) Find the root mean square error
import numpy as np
rms_error=np.sqrt(ms_error)
print("root mean square error=",rms_error)

#1)Bild a simple linear model where dependent variable is "Monthly charges" and independent variable is "tenure"
from sklearn.linear_model import LogisticRegression
LoR=LogisticRegression() #give handle name to our algorithm 
x=pd.DataFrame(data['MonthlyCharges']) #Load input data x
y=data['Churn'] #Load output data y

#2) Data is devided into 70% train and remianing 30% as test data
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.70,random_state=0) 
#3) Train our LoR neural network with training data
LoR.fit(xtrain, ytrain)
#Now our Logistic Regression(LoR) model is trained with 70% data

ypred=LoR.predict(xtest)

ypred

ytest

#There are some errors in the prediction of churn being "Yes" or "No"
#Taking RMS error will not give proper quatitative measure for binary (Yes/No) output prediction
#Hence we use confusion matrix to measure the amount of error 
from sklearn.metrics import confusion_matrix, accuracy_score
M=confusion_matrix(ypred,ytest)
print("Confusion Matrix\n",M)
#M_11=TP, M_12=FP
#M_21=FN, M_22=TN
S=accuracy_score(ypred,ytest)
print("Accuracy of Prediction=",S)



from sklearn.linear_model import LogisticRegression
LoR=LogisticRegression() #give handle name to our algorithm 
x=pd.DataFrame(data.loc[:,['MonthlyCharges','tenure']])
y=data['Churn']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.70,random_state=0) 
LoR.fit(xtrain, ytrain)
ypred=LoR.predict(xtest)
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score
M=confusion_matrix(ypred,ytest)
print("Confusion Matrix\n",M)
#M_11=TP, M_12=FP
#M_21=FN, M_22=TN
S=accuracy_score(ypred,ytest)
print("Accuracy of Prediction=",S)

#C 3.1) Add extra extra parameter at input and check if accuracy increses 
x=pd.DataFrame(data.loc[:,['MonthlyCharges','tenure','SeniorCitizen']])
y=data['Churn']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.70,random_state=0) 
LoR.fit(xtrain, ytrain)
ypred=LoR.predict(xtest)
from sklearn.metrics import confusion_matrix, accuracy_score
M=confusion_matrix(ypred,ytest)
print("Confusion Matrix\n",M)
#M_11=TP, M_12=FP
#M_21=FN, M_22=TN
S=accuracy_score(ypred,ytest)
print("Accuracy of Prediction=",S)

from sklearn.tree import DecisionTreeClassifier
DecisionTree=DecisionTreeClassifier()
x=pd.DataFrame(data['tenure'])
y=data['Churn']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
DecisionTree.fit(xtrain,ytrain)
ypred=DecisionTree.predict(xtest)

from sklearn.metrics import confusion_matrix, accuracy_score
M=confusion_matrix(ypred,ytest)
print("Confusion Matrix\n",M)
#M_11=TP, M_12=FP
#M_21=FN, M_22=TN
S=accuracy_score(ypred,ytest)
print("Accuracy of Prediction=",S)

from sklearn.ensemble import RandomForestClassifier
RandomForest=RandomForestClassifier()
x=pd.DataFrame(data.loc[:,['MonthlyCharges','tenure']])
y=data['Churn']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
RandomForest.fit(xtrain,ytrain)
ypred=RandomForest.predict(xtest)
from sklearn.metrics import confusion_matrix, accuracy_score
M=confusion_matrix(ypred,ytest)
print("Confusion Matrix\n",M)
#M_11=TP, M_12=FP
#M_21=FN, M_22=TN
S=accuracy_score(ypred,ytest)
print("Accuracy of Prediction=",S)

#Add extra extra parameter at input and check if accuracy increses 
from sklearn.ensemble import RandomForestClassifier
RandomForest=RandomForestClassifier()
x=pd.DataFrame(data.loc[:,['MonthlyCharges','tenure', 'SeniorCitizen']])
y=data['Churn']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
RandomForest.fit(xtrain,ytrain)
ypred=RandomForest.predict(xtest)
from sklearn.metrics import confusion_matrix, accuracy_score
M=confusion_matrix(ypred,ytest)
print("Confusion Matrix\n",M)
#M_11=TP, M_12=FP
#M_21=FN, M_22=TN
S=accuracy_score(ypred,ytest)
print("Accuracy of Prediction=",S)
