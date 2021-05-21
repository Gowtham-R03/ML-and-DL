### POLYNOMIAL REGRESSION COVID 19 #####

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt
#### LOAD DATA ####

data = pd.read_csv('indiacoronacases.csv')

data =  data[['id','cases']]
print('-'*30);print(" HEAD ");print('-'*30)
print(data.head())

##### PREPARE DATA ######
print('-'*30);print(" PREPARE DATA ");print('-'*30)

x = np.array(data['id']).reshape(-1,1)
y = np.array(data['cases']).reshape(-1,1)

plt.plot(y,'-m')

polyFea = PolynomialFeatures(degree = 9) #degree for order of quation 
x = polyFea.fit_transform(x)

##### TRAINING OF DATA #####

print('-'*30);print(" PREPARE DATA ");print('-'*30)
#x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.2)

#here the data is less so we can make model with x and y itself

model = linear_model.LinearRegression()
model.fit(x,y)

accuracy = model.score(x,y)
print('Accuracy:',round(accuracy*100,3),'%')

y0 = model.predict(x)


##### PREDICTION ####
days = 1
print('-'*30);print(" PREDICTION ");print('-'*30)
print(f'Prediction cases after {days} days:',end='')
print(round(int(model.predict(polyFea.fit_transform([[456+days]])))/1000000,5),'million')

x1 = np.array(list(range(1,456+days))).reshape(-1,1)
y1 = model.predict(polyFea.fit_transform(x1))
plt.plot(y1,'--r')
plt.plot(y0,'--b')
plt.show()