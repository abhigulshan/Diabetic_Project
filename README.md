# Diabetic_Project
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
diabetes_data = pd.read_csv('D:\Downloads\pr\diabetes.csv')
#PRINT TOP 5 ROWS
diabetes_data.head(5)
diabetes_data.describe
#value counts of outcomes like how much patient are efected and how much paitent are not
#level 0 is non diabitec and 1 is diabeti
diabetes_data["Outcome"].value_counts()
# 0-for non-diabetic and 1-for diabetic
# 1-for diabetic and 0-for non-diabetic
diabetes_data.groupby('Outcome').mean()
#seprating the data and levels 
x = diabetes_data.drop(columns='Outcome', axis=1)   
y = diabetes_data['Outcome']
print(x)
print(y)
#data standardization
scaler= StandardScaler()
scaler.fit(x)
#splitting the data into training and testing data
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
standersized_data = scaler.transform(x)
standersized_data
x=standersized_data
y=diabetes_data['Outcome']
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y, random_state=2)
print(x.shape,x_train.shape,x_test.shape)
classifier = svm.SVC(kernel='linear')
#training the support vector machine classifier 
classifier.fit(x_train, y_train)    
#Accuracy score on the training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
#Accuracy Score
print("Accuracy score of the training data : ", training_data_accuracy) 
#Accuracy Score on the test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
#Accuracy Score
print("Accuracy score of the test  data : ", test_data_accuracy) 
input_data = (4,110,92,0,0,37.6,0.191,30)
#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
#standardize the input data
std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)

if(prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
input_data = (5,166,72,19,175,25.8,0.587,51)
#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
#standardize the input data
std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)

if(prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
      
