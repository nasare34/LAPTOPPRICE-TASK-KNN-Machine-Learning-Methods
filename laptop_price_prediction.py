# -*- coding: utf-8 -*-
"""Laptop Price Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NQBDdCipKXueqa1jR5y-Y_FDruhKSdJZ
"""

#importing the appropriate libraries for the machine learning task
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

from google.colab import drive
drive.mount('/content/drive')

# Provide the path to your TSV file in Google Drive
file_path = '/content/drive/MyDrive/laptopprices.tsv'

# Reading the TSV file
training_data = pd.read_csv(file_path, delimiter='\t')

# I am Displaying the first few rows of the DataFrame
print(training_data.head())

# Confirming the quality of the datasets
training_data.describe()

# Splitting the DataFrame into input features (X) and target variable (y)
X = training_data.iloc[:, :-1].values  # X contains all columns except the last one
y = training_data.iloc[:, -1].values   # y contains only the last column, which is the target variable

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20,random_state=0)

# Convert X_train and X_test to NumPy arrays

X_train = np.array(X_train)
X_test = np.array(X_test)

# Encode categorical features in X_train and X_test using LabelEncoder to change the non-numerical values to values
#such as the 'CPU', 'GPU', 'RAMType', 'SSD' to enable computation to take place.!pip install scikit-learn
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for i in range(0, len(X_train[0])):
    if type(X_train[0][i]) == str:
        X_train[:, i] = le.fit_transform(X_train[:, i])
    if type(X_test[0][i]) == str:
        X_test[:, i] = le.transform(X_test[:, i])

#Checking the x dataset after the labeling encoder
print(X_train)
print(X_test)

# Standardize features in X_train and X_test
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize a K-Nearest Neighbors classifier

classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 2)

# Model training and Prediction with the X_test
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:,1]

import matplotlib.pyplot as plt
import seaborn as sns

# Check if y_test and y_pred are defined
if 'y_test' not in globals() or 'y_pred' not in globals():
    raise ValueError("y_test and y_pred must be defined before calculating the confusion matrix.")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

#Accuracy
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

# Classification report
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

# Generate classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Convert the report to a DataFrame
df_report = pd.DataFrame(report).transpose()

# Plotting
plt.figure(figsize=(10, 5))
sns.heatmap(df_report.iloc[:-1, :].astype(float), annot=True, cmap="Blues")
plt.title('Classification Report')
plt.show()

# Making some real predictions
new_prediction = classifier.predict(sc.transform(np.array([[-0.25775, 0.725608, 0.0187845, -0.206688, -0.745665, 1.17924, 0.840852, -2.97799, 0.573857]])))
new_prediction_proba = classifier.predict_proba(sc.transform(np.array([[-0.25775, 0.725608, 0.0187845, -0.206688, -0.745665, 1.17924, 0.840852, -2.97799, 0.573857]])))[:, 1]

print(new_prediction)
print(new_prediction_proba)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Save the classifier
model_file = "classifier.pickle"
with open(model_file, 'wb') as file:
    pickle.dump(classifier, file)

# Save the scaler
scaler_file = "sc.pickle"
with open(scaler_file, 'wb') as file:
    pickle.dump(sc, file)

#  Save the X_test and y_test data for future use
testdata_file = "testdata.pickle"
with open(testdata_file, 'wb') as file:
    pickle.dump((X_test, y_test), file)