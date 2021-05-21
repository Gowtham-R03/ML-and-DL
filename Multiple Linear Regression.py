# Multiple linear regression
import pandas as pd
import sklearn
from sklearn import preprocessing,linear_model
import numpy as np


#### LOAD DATA ###
print('-'*30);print('IMPORTING DATA');print('-'*30)
data = pd.read_csv('houses_to_rent.csv', sep = ',')

data = data[['city','rooms','bathroom','parking spaces','fire insurance',
         'furniture','rent amount']]



### PROCESSING OUR DATA ###
# to remove unwanted and make strings to ineger

# in rent amount to remove R$ we doing process and also remove comma
data['rent amount'] = data['rent amount'].map(lambda i: int(i[2:].replace(',' ,'')))
data['fire insurance'] = data['fire insurance'].map(lambda i: int(i[2:]))

# for to encode furnised-0 not furnides -1
le = preprocessing.LabelEncoder()
data['furniture'] = le.fit_transform(data['furniture'] )

print('-'*30);print('CHECKING NULL DATA');print('-'*30)
# to check nan in our data
print(data.isnull().sum())
#   if null present to drop nan data
# data = data.dropna() #nan data will be removed
# print(data.isnull().sum())

print('-'*30);print('HEAD');print('-'*30)
print(data.head())

##### SPLIT DATA #####

print('-'*30);print('SPLIT DATA');print('-'*30)

x = np.array(data.drop(['rent amount'],1))
y = np.array(data['rent amount'])
print('X',x.shape)
print('Y',y.shape)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.2)
print('XTrain',x_train.shape)
print('XTest',x_test.shape)

#train our data
print('-'*30);print('TRAINING');print('-'*30)

model = linear_model.LinearRegression()
model.fit(x_train,y_train) # to get best fit from train
# now model is created 
print('Coefficient',model.coef_) #for multiple linear regression we have many coefficient
print('Intersecpt',model.intercept_)

accuracy = model.score(x_test,y_test)
print('accuracy',round(accuracy*100,3),'%')

##### EVALUATION #####
print('-'*30);print('MANUAL TESTING');print('-'*30)

testvals = model.predict(x_test)
print(testvals.shape)
error =[]
for i,testval in enumerate(testvals):  #enumurate used for instead fr count
    error.append(y_test[i]-testval)
    print('Actual:{}, Prediction:{}, Error:{}'.format(y_test[1],int(testvals),int(error[i])))
    