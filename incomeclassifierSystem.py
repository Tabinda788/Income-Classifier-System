import os
os.chdir('C:\\Users\\DELL\\Desktop\\nptl online course\\python for data science\\week 4')
# To work with dataframes
import pandas as pd

# To Perform Numerical Operations
import numpy as np

# To visualize data
import seaborn as sns

# To partition data
from sklearn.model_selection import train_test_split

# Importing libraries for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing performance matrics-accuracy score and confusoion matrices
from sklearn.metrics import accuracy_score,confusion_matrix

#importing data
data_income=pd.read_csv('income.csv')

# Creating a copy of original data
data = data_income.copy()

"""
# Exploratory Data analysis:

#1.Gettin to know the data
#2.Data preprocessing(Missing values)
#3.Cross tables and Data visualisation
"""
# Getting to know the data
#Checking Missing values
print(data.info())

data.isnull()

print('Data columns with null values: \n',data.isnull().sum())
#**** There are nomissing values!!!!!
# Summary of the numerical variables.
summary_num = data.describe()

print(summary_num)

#Summary of catagorical variables
summary_cate = data.describe(include = "O")

print(summary_cate)
data.info()
# Frequency of each catagory
data['JobType'].value_counts()
data['occupation'].value_counts()

#Checking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
"""
Go back and read the data by including  "na_values[ ?] to consider ' ?' as nan"
"""
data= pd.read_csv('income.csv',na_values=[" ?"])

#Data preprocessing
data.isnull().sum()

missing = data[data.isnull().any(axis=1)]
# Axis =>1to consider atleast one column value is missing

""" Points to note:
1. Missing values in jobType  = 1809
2. Missing value in occupation = 1816
3. There are 1809 rows where2 specific columns
i.e. occupation and jobtype have missing values
4. (1816-1809)=7 => you still have occupation unfilled for
these 7rows because jobtype is never worked
"""
data2=data.dropna(axis=0)
correlation=data2.corr()
data2.info()
#Cross tables and datavisualisation
# Extracting the column names
data2.columns

#Gender proportion table:
gender=pd.crosstab(index = data2["gender"],
                   columns = 'count',
                   normalize= True)
print(gender)

#Gender vs Salary Status
gender_salstat=pd.crosstab(index =data2["gender"],
                           columns= data2["SalStat"],
                           margins=True,
                           normalize='index' #Include row proportion =1
                        )
print(gender_salstat)

# Frequency distribution of salary status
SalStat=sns.countplot(data2['SalStat'])
data2.info()
"""
75% of peoples salary status is<=500000
25% of peoples salary status is>500000
"""
sns.distplot(data2['age'],bins=10,kde=False)
# People with age 20-45 are high in frequency
#Box plot Age vs salarystatus
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()

salstat_JobType=pd.crosstab(index =data2["JobType"],
                           columns= data2["SalStat"],
                           margins=True,
                           normalize='index' #Include row proportion =1
                        )
print(salstat_JobType)

salstat_EdType=pd.crosstab(index =data2["EdType"],
                           columns= data2["SalStat"],
                           margins=True,
                           normalize='index' #Include row proportion =1
                        )
print(salstat_EdType*100)

salstat_occupation=pd.crosstab(index =data2["occupation"],
                           columns= data2["SalStat"],
                           margins=True,
                           normalize='index' #Include row proportion =1
                        )
print(salstat_occupation*100)
 
sns.distplot(data2['capitalgain'],bins=10,kde=False)

sns.distplot(data2['capitalloss'],bins=10,kde=False)

sns.boxplot('SalStat','hoursperweek',data=data2)
data2.head()
#Reindexing the salary ststus name to 0 and 1
#data2[alStat'] = data2['SalStat'].map({'less than or equal to 50,000':0,'greater than 50,000':1})
data2['SalStat']=pd.factorize(data2.SalStat)[0]
print(data2['SalStat'])

new_data=pd.get_dummies(data2, drop_first=True)
#Sorting the column names
columns_list=list(new_data.columns)
print(columns_list)
#Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)



#data2.info()
#data2.dtypes

#data2.loc[:,'SalStat'].astype('object')
#data2.info()
#data2.loc[:,'SalStat']=data2.loc[:,'SalStat'].map({'less than or equal to 50,000':0,'greater than 50,000':1})
#data2.info()
#print(data2['SalStat'])

#Storing the otput values in y
y=new_data['SalStat'].values
print(y)


#Sorting the values from input features
x=new_data[features].values
print(x)
#Splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

#Make an instance of model
logistic = LogisticRegression()


#Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#Prediction for test_data
prediction=logistic.predict(test_x)
print(prediction)

#Confusion Matirix
confusion_matrix = confusion_matrix(test_y,prediction)
print(confusion_matrix)

#Calculating the accuracy
accuracy_score = accuracy_score(test_y,prediction)
print(accuracy_score)
#Printing the misclassified values from prediction
print('Misclassified samples: %d' %(test_y != prediction).sum())


data2['SalStat']=pd.factorize(data2.SalStat)[0]
print(data2['SalStat'])
data2.head()
cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)

#Get dummes for all catagorical varibles
new_data=pd.get_dummies(new_data,drop_first=True)

#Sorting the column names
columns_list=list(new_data.columns)
print(columns_list)
#Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)
#Storing the otput values in y
y=new_data['SalStat'].values
print(y)


#Sorting the values from input features
x=new_data[features].values
print(x)
#Splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

#Make an instance of model
logistic = LogisticRegression()

#Fitting the values for x and y
logistic.fit(train_x,train_y)


#Prediction for test_data
prediction=logistic.predict(test_x)
print(prediction)



#Calculating the accuracy
accuracy_score = accuracy_score(test_y, prediction)
print(accuracy_score)

#Printing the misclassified values from prediction
print('Misclassified samples: %d' %(test_y != prediction).sum())


#importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

#import libarary for plotting
import matplotlib.pyplot as plt

#Store the KNN classifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)

#Fitting the values of x and y
KNN_classifier.fit(train_x, train_y)
#Predicting the test variables with model
prediction=KNN_classifier.predict(test_x)

#performance matrix check
confusion_matrix = confusion_matrix(test_y, prediction)
print("\t","Predicted values")
print("original values","\n",confusion_matrix)
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

#Printing the misclassified values from prediction
print('Misclassified samples: %d' %(test_y != prediction).sum())

#Effect of k values on classifier
Misclassified_Sample = []
 for i in range( 1, 20):
     knn= KNeighborsClassifier(n_neighbors=i)
     knn.fit(train_x,train_y)
     pred_i=knn.predict(test_x)
     Misclassified_Sample.append((test_y != pred_i).sum())
     
 print(Misclassified_Sample)
 
 
 
 
 























































































