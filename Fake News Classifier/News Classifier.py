#News Classifier
#Goal: To create a text analysis model that can classify if a news text is fake or real 
#Important modules: sklearn, pandas and numpy 

import pandas as pd 
import numpy as np 
import sklearn
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

path = "/Users/duanemadziva/Documents/_ Print (Hello World)/Learning Python/PythonVS/Data/"
news_data = pd.read_csv(path+"news.csv"); news_data.head()
#check missing data 
news_data.isnull().sum()

#get labels/ target variable 
target = news_data["label"]; target.head()
text = news_data["text"]

#create model 
#split dataset
x_train, x_test, y_train, y_test = train_test_split(text, target, random_state = 7, test_size = 0.25)

#Load TfidfVectorizer, fit and transform x_train and x_test
tdidf_object = TfidfVectorizer(stop_words="english", max_df=0.75)  #if a word appears with a frequency 0.7 in each document it will be discarded -"stop word"
x_train = tdidf_object.fit_transform(x_train)
x_test = tdidf_object.transform(x_test)

#Load Passive Aggressive Classifier, fits the model, and then create prediction 
news_model = PassiveAggressiveClassifier(max_iter=50)
news_model.fit(x_train, y_train)

predictions = news_model.predict(x_test)
#evaluate model
accuracy = accuracy_score(predictions, y_test)
print("Accuracy score: ", format(accuracy, ".2%"))

confusion_matrix(y_test, predictions, labels = ["FAKE", "REAL"])

#test model on a fake news articles - Fake News Articles from https://www.snopes.com/fact-check/
news_article = pd.read_csv(path+"news_articles.csv")
news_article = tdidf_object.transform(news_article["Article"])
#make predictions on the 3 news articles, which we know all are fake 
prediction = news_model.predict(news_article)  #model correctly identifies all 3 articles as fake 
print("Expected number of fake articles: 3. Model predicted ", prediction.tolist().count("FAKE"), " fake articles")




 
