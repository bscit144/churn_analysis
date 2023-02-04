import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data=pd.read_csv(r"C:\Users\Jose\Desktop\churn_modelling.csv")

print(data)

data.describe()

data.info()

data.isnull()

sns.countplot(x='Gender',hue='Geography',data=data)

sns.countplot(x='Exited',hue='Gender',data=data)

sns.boxplot(x='Exited',y='Geography',data=data)

geography=pd.get_dummies(data['Geography'],drop_first=True)
geography.head()

data.head()

gender=pd.get_dummies(data['Gender'],drop_first=True)
gender.head()

data.head()

data=pd.concat([data,gender,geography],axis=1)

data.head()

data.drop(['Surname','Geography','Gender'],axis=1,inplace=True)

data.head()

x_train,x_test,y_train,y_test=train_test_split(data.drop('Exited',axis=1),data['Exited'],test_size=0.30,random_state=101)

logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
prediction=logmodel.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))

         **********
/*EXPLANATION OF FINDINGS/*
-first we have to import libraries necessary for logistics regression.then load the data using the pandas library.
-We have explore the data by checking whether there is null values,the datatypes that makes the data,check the number of columns androws the data has,
-After exploring the data,we perform data wrangling where we replace null values
-Through analysis,we find that ,there is large number of males captured in the dataset in all the three geographical countries that female.
The analysis also shows;
        large number of females exited as compare to their male counterparts
        from the show of boxplot,people from germany exited in large number as compared to those from other two countries captured in the dataset
-Since logistic regression only uses numerical data for prediction, we have converted the string values to categorical values inform of 1 and 0,in this 
case, the data for Geography,and Gender have been converted into 1 and 0 where for gender,1 rep male and 0 for females
-