import pandas as pd, numpy as np, re
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Loading data from json file
data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines = True)
# Relacing special symbols and digits in headline column
# re stands for Regular Expression
data['headline'] = data['headline'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
# getting features and labels
features = data['headline']
labels = data['is_sarcastic']
# Stemming our data
ps = PorterStemmer()
features = features.apply(lambda x: x.split())
features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))
# vectorizing the data with maximum of 5000 features
from sklearn.feature_extraction.text import TfidfVectorizer
# tv = TfidfVectorizer(max_features = 5000, stop_words='english',min_df = 5, ngram_range=(2,2)) ## 65.54 and 63.92
# tv = TfidfVectorizer(max_features = 5000, stop_words='english', ngram_range=(2,2)) ## 74.07 and 67.51
# tv = TfidfVectorizer(max_features = 5000, stop_words='english',min_df = 5) ## 87.97 and 79.57
# tv = TfidfVectorizer(max_features = 5000,min_df = 5, ngram_range=(2,2)) ## 81.67 and 72.23
# tv = TfidfVectorizer(max_features = 5000,min_df = 5, ngram_range=(1,2)) ## 91.22 and 82.71
tv = TfidfVectorizer(max_features = 5000,min_df = 5, ngram_range=(1,3)) ## 91.15 and 83.01

features = list(features)
features = tv.fit_transform(features).toarray()
# getting training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .05, random_state = 0)
# Using linear support vector classifier
lsvc = LinearSVC()
# training the model
lsvc.fit(features_train, labels_train)
# getting the score of train and test data
print(lsvc.score(features_train, labels_train)) # 90.93
print(lsvc.score(features_test, labels_test))   # 83.75