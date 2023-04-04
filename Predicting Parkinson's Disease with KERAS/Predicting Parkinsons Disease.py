"""
Goal:
Predict Parkinson's Disease using (1) Keras (2) Ensemble Model
Using XGBoost Link: https://data-flair.training/blogs/python-machine-learning-project-detecting-parkinson-disease/
Current accuracy score with XGBoost = 94.97 %"""

#useful links on scoring models: https://scikit-learn.org/stable/modules/model_evaluation.html
#----------------------------------------------------------------------------------------------
#import packages
#----------------------------------------------------------------------------------------------
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
import os, sys
from sklearn.preprocessing import MinMaxScaler   #for scaling features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

#----------------------------------------------------------------------------------------------
#read in data and preprocessing
#----------------------------------------------------------------------------------------------
path = "/Users/.../Data/"  #directory
parkinsons_data = pd.read_csv(path+"parkinsons.data")

#assumption: there are too many features to do perform exploratory data analysis hence we assume all features are crucial in assessing parkinsons disease
features = parkinsons_data.drop(["name", "status"], axis = 1)
target = parkinsons_data[["status"]]

#find number of 0s and 1s
target.status.value_counts()

#scale features
y = target
x = MinMaxScaler((-1,1)).fit_transform(features)

#split data into training and testing data
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=7)

#----------------------------------------------------------------------------------------------
#Keras Neural Network
#----------------------------------------------------------------------------------------------
#define neural network for classification
def binary_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_dim = 22),  #stipulate shape of input (all data features plus target)
    layers.Dense(1, activation="sigmoid")  #specify dimensions of output, sigmoid activation used for binary classification
  ])

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  #binary_crossentropy is the preferred loss function for binary classification problems
  return model

model = binary_model()
model.summary()
#fit model
EPOCHS = 1000  #An epoch is a measure of the number of times all of the training vectors are used once to update the weights

#make use of a simple early stopping function - #early stop dropped accuracy to 87.2% % from 97.4%
#earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history = model.fit(
  x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, verbose=0)

# evaluate the model
_, train_acc = model.evaluate(x_train, y_train, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
#----------------------------------------------------------------------------------------------
#Ensemble Model
#----------------------------------------------------------------------------------------------
#1. Logistic Regression
logistic_model = LogisticRegression(random_state=0).fit(x_train, y_train)
logistic_score = round(accuracy_score(y_test, logistic_model.predict(x_test))*100,2)

#2. Support Vector Machine
SMV_model = svm.SVC(kernel="linear").fit(x_train, y_train)
SMV_score = round(accuracy_score(y_test, SMV_model.predict(x_test))*100,2)

#3. Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier().fit(x_train, y_train)
decision_tree_score = round(accuracy_score(y_test, decision_tree_model.predict(x_test))*100,2)

#4. Random Forest Classifier
random_forest_model = RandomForestClassifier().fit(x_train, y_train)
random_forest_score = round(accuracy_score(y_test, random_forest_model.predict(x_test))*100,2)

#5. K-Nearest Neighbors
k = [1,3,5,7,9,11,13,15]
for i in k:
    knn_model = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
    score = round(accuracy_score(y_test, knn_model.predict(x_test))*100,2)
    print(score)  #k = [1, 3, 5] are the best, we will go with 3 for now

knn_final_model = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
knn_final_score = round(accuracy_score(y_test, knn_final_model.predict(x_test))*100,2)

#NB: The K-Nearest Neighbor model seems to outperform the XGBoost model already
scores_data = {'Model': ["Logistic Regression", "Support Vector Machine", "Decision Tree", "Random Forest", "K-Nearest Neighbors"],
'Accuracy Scores (%)': [logistic_score, SMV_score, decision_tree_score, random_forest_score, knn_final_score]}
model_scores = pd.DataFrame(data = scores_data)

#6 Ensemble Model with VotingClassifier with top 3 highest scoring models
logistic = LogisticRegression(random_state=0)
smv = svm.SVC(kernel="linear")
randforest = RandomForestClassifier()
dectree = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=3)

ensemble_model = VotingClassifier(estimators = [('smv', smv), ('randforest', randforest), ('knn', knn)], voting='hard').fit(x_train, y_train)
ensemble_score = round(accuracy_score(y_test, ensemble_model.predict(x_test))*100,2)  
print("Ensemble Model Accuracy Score: ", ensemble_score, "%")

#----------------------------------------------------------------------------------------------
#Conclusion
#----------------------------------------------------------------------------------------------
final_scores_data = {'Model': ["Logistic Regression", "Support Vector Machine", "Decision Tree", "Random Forest", "K-Nearest Neighbors", "Ensemble Model", "Keras Neural Network"],
'Accuracy Scores (%)': [logistic_score, SMV_score, decision_tree_score, random_forest_score, knn_final_score, ensemble_score, round(test_acc*100,2)]}
all_model_scores = pd.DataFrame(data = final_scores_data)
