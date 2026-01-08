#house prize prediction

#supervised learning is of two types
#1. classification  : predicts a class eg yes/no
#2. regression : predicts a value eg rate

#steps
#step 1 : data preprocessing
#step 2 : data analysis
#step 3 : train test split
#step 4 : XGBoosRegressor
#step 5 : Evaluation


#importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#importing the Boston House Price Dataset
house_price_data = pd.read_csv('/content/housing.csv')

#print first five rows of the data sheet
print(house_price_data)

#check null values in various columns
house_price_data.isnull().sum()

#finding mathematicals features of datasheet
house_price_data.describe()

#finding correlation in various attributes
#positive correlation : values are directly proportional
correlation = house_price_data.corr()
print(correlation)

#constructing a heat map
plt.figure(figsize=(5,5))
sns.heatmap(correlation, cbar=True, fmt='.1f', annot = True, annot_kws={'size':8}, cmap ='Blues' )
#light color is negative correlation

#dividing the dataset in two parts
X =  house_price_data.drop(['MEDV'], axis=1)
Y = house_price_data['MEDV']
print(X)
print(Y)

#train test split data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2, shuffle =True)

#model training
model = XGBRegressor()
model.fit(X_train, Y_train)

#model evaluation
training_data_prediction = model.predict(X_train)
print(training_data_prediction)
score1 = metrics.r2_score(Y_train, training_data_prediction)
print(score1)
test_data_prediction = model.predict(X_test)
score2 = metrics.r2_score(Y_test , test_data_prediction)
print(score2)

#scatter plot
plt.scatter(Y_train, training_data_prediction)
plt.xlabel('Actual Prizes')
plt.ylabel('Predicted Prizes')
plt.show()

#changing the input data to numpy array

input_data = (6.575,4.98,15.3)
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
