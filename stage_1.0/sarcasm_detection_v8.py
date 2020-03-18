## v8: remove parent_comment

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
df = pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
df = df.rename(columns={df.columns[0]:"is_sarcastic", df.columns[1]:"comment", df.columns[2]:"parent_comment"})
data = df[0:15000]



FNAME = "output_v8_data_size={}.txt".format(len(data))
output_file = open(FNAME,'w')
output_file.write("Dataset size:{}\n".format(len(data)))

# Relacing special symbols and digits in comment column
# re stands for Regular Expression

data['comment'] = data['comment'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
# getting features and labels
features = data['comment']

comments_file=open('comments_v6.txt','w')
for i in range(len(features)):
    comments_file.write("{}    {}\n".format(i,features[i]))
comments_file.close()

labels = data['is_sarcastic']
# Stemming our data
ps = PorterStemmer()
features = features.apply(lambda x: x.split())
features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))
# vectorizing the data with maximum of 5000 features
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features = 5000,min_df = 5, ngram_range=(1,2))
features = list(features)
features = tv.fit_transform(features).toarray()
# getting training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .05, random_state = 0)
output_file.write("x_train size:{}\n".format(len(features_train)))
output_file.write("x_test size:{}\n".format(len(features_test)))
# Using linear support vector classifier
lsvc = LinearSVC()
# training the model
lsvc.fit(features_train, labels_train)
# getting the score of train and test data
output_file.write("\nModel: LinearSVC\n")
output_file.write("train score: {}\n".format(lsvc.score(features_train, labels_train))) 
output_file.write("test score: {}\n".format(lsvc.score(features_test, labels_test)))   
# model 2:-
# Using Gaussuan Naive Bayes
gnb = GaussianNB()
gnb.fit(features_train, labels_train)
output_file.write("Model: GaussianNB\n")
output_file.write("train score: {}\n".format(gnb.score(features_train, labels_train)))  
output_file.write("test score: {}\n".format(gnb.score(features_test, labels_test)))   
# model 3:-
# Logistic Regression
lr = LogisticRegression()
lr.fit(features_train, labels_train)
output_file.write("Model: LogisticRegression\n")
output_file.write("train score: {}\n".format(lr.score(features_train, labels_train)))  
output_file.write("test score: {}\n".format(lr.score(features_test, labels_test)))     
# model 4:-
# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 10, random_state = 0)
rfc.fit(features_train, labels_train)
output_file.write("Model: RandomForestClassifier\n")
output_file.write("train score: {}\n".format(rfc.score(features_train, labels_train)))  
output_file.write("test score: {}\n".format(rfc.score(features_test, labels_test)))    