# MACHINE LEARNING MODEL

# import libararies
import numpy as np
import pandas as pd

#importing dataset (iris.csv)

data = pd.read_csv('Iris.csv')

# separate dependent and independent element from dataset
x = data.iloc[:,1:-1] # independent variable
y = data.iloc[:,-1] # dependent variable

# ML take only numeric value(numbers)
# example: male and female
# male-0 female -1 ;This is lable encoding bcz we assin 0 for male and 1 for female
# One Hot Encoding; it makes column of unique element in label encoding
# 1 2 ; 1 2 3
# 1 0   1 0 0
# 0 1   0 1 0
#       0 0 1

# for both label and hot encoding we fun called LabelBinarizer()
from sklearn.preprocessing import LabelBinarizer
#making object of labelbinarixer
le = LabelBinarizer()
# transform the y values into numbet or binay
y = le.fit_transform(y) 

# For ML model first we need to train ML model with some data that is train dataset
# after modelling we need to test our model with different dataset bcz our model is working or not that is test dataset
# 1/3 rd of data will taken for test remain will taken for train.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

#regression = Classifier
#Creating classifier object that will train the model  based on training model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)

# Now will predict using test sets
y_pred = classifier.predict(x_test)
# y predection = y test

# to calculate accuracy of our model comparing manually we can use from sklearn

from sklearn.metrics import confusion_matrix,accuracy_score
#confusion matrix store matrix like 3x3 matrix other than daigonal it has output
cm = confusion_matrix(y_test.argmax(axis = 1),y_pred.argmax(axis =1))
# argmax used for takin specific index in which has(1) in the dataset column
acc = accuracy_score(y_test.argmax(axis=1),y_pred.argmax(axis=1))
#from acc value we can get our model accuracy

#Save ML model
import pickle

# Save the trained model as a pickle string.
Pkl_Filename = "ML_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(classifier, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    ML_Model = pickle.load(file)

ML_Model
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = ML_Model.score(x_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
y_pred = ML_Model.predict(x_test)  

y_pred
