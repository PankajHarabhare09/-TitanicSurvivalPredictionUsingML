import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Load The Titanic Dataset
Titanic = pd.read_csv("titanic.csv")

# Inspect The Dataset
print("------> Display The First 5 Rows Of The Dataset <------")
print(Titanic.head(),"\n")
print("------> Display The Last 5 Rows Of The Dataset <------")
print(Titanic.tail(),"\n")
print("------> Display the number of rows and columns in theDataset <------")
print(Titanic.shape,"\n")
print("------> Display the column names in the dataset <------")
print(Titanic.columns,"\n")
print("------> Display the summary of the dataset <------")
print(Titanic.info(),"\n")
print("------> Display the statistical summary of the numerical columns <------")
print(Titanic.describe(),"\n")


# #Data Cleaning
print("------> Check for Missing Values in the Dataset <------")
print(Titanic.isnull().sum(),"\n")
Titanic.drop(columns=["Cabin"], inplace=True)
Titanic["Age"].fillna(Titanic["Age"].median() , inplace=True)
Titanic.dropna(inplace=True)
print("------> Check for Missing Values After Cleaning <------")
print(Titanic.isnull().sum(),"\n")

# #Data Selection and Filtering
print("Age \n",Titanic["Age"])
print("Name And Age \n" , Titanic[["Name", "Age"]])
print("Age Greater Than 30 \n" , Titanic[Titanic["Age"] > 30])
print( "Female \n",Titanic[Titanic["Sex"] == "female"])

print("Mean \n" , np.mean(Titanic["Age"]))
print("Median \n" ,np.median(Titanic["Age"]))

print(Titanic.describe())

print("------> Display the number of survivors and non-survivors <------")
print(Titanic.groupby('Sex')["Survived"].count())

print("------> Display the survival rate by passenger class <------")
print(Titanic.groupby('Pclass')['Survived'].count())

print("------> Display the survival rate by embarkation port <------")
print(Titanic.groupby('Embarked')['Survived'].count())

plt.figure()
Titanic["Survived"].value_counts().plot(kind = "bar")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("number of passengers")
plt.show()

plt.figure()
plt.hist(Titanic["Age"] , bins = 25 , edgecolor = "black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure()
Titanic.groupby("Pclass")["Survived"].count().plot(kind = "bar")
plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

print("------> converting categorical variables to numerical <------")
Titanic["Sex"] = Titanic["Sex"].map({"male": 0, "female": 1})
Titanic["Embarked"] = Titanic["Embarked"].map({"S": 0, "C": 1, "Q": 2})
print(Titanic.head())


print("------> Creating a new feature <------")
Titanic["FamilySize"] = Titanic["SibSp"] + Titanic["Parch"]
print(Titanic[["FamilySize"]].head(),"\n")

print("------> Prepare the data for machine learning <------")
features = ["Pclass", "Sex" , "Age" , "Fare" , "FamilySize"]

X = Titanic[features]
y = Titanic["Survived"]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)

print("------> Training a machine learning model <------")
model = LogisticRegression(max_iter=200)
model.fit(X_train , y_train)

predictions = model.predict(X_test)

print("------> Accuracy of the model <------")
accuracy = model.score(X_test , y_test)
print(accuracy)
