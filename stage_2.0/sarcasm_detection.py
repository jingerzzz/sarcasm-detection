

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
train_df = pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
train_df = train_df.rename(columns={train_df.columns[0]:"is_sarcastic", train_df.columns[1]:"comment", train_df.columns[2]:"parent_comment"})
test_df = pd.read_csv('labeled_test.csv',delimiter=',',encoding='utf-8')
test_df = test_df.rename(columns={test_df.columns[0]:"id", test_df.columns[1]:"comment", test_df.columns[2]:"parent_comment",test_df.columns[3]:"is_sarcastic"})
train_df = train_df.dropna()
test_df = test_df.dropna()

FNAME = "output_data_size={}.txt".format(len(train_df)+len(test_df))
output_file = open(FNAME,'w')
output_file.write("Dataset size:{}\n".format(len(train_df)+len(test_df)))

# Relacing special symbols and digits in comment column
# re stands for Regular Expression

train_df['comment'] = train_df['comment'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
# test_df['comment'] = test_df['comment'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
# getting features and labels
features_train = train_df['comment']
features_test = test_df['comment']

# comments_file=open('comments_v5.txt','w')
# for i in range(len(features)):
#     comments_file.write("{}    {}\n".format(i,features[i]))
# comments_file.close()

labels_train = train_df['is_sarcastic']
labels_test = test_df['is_sarcastic']
# Stemming our data
ps = PorterStemmer()
features_train = features_train.apply(lambda x: x.split())
features_train = features_train.apply(lambda x : ' '.join([ps.stem(word) for word in x]))
features_test = features_test.apply(lambda x: x.split())
features_test = features_test.apply(lambda x : ' '.join([ps.stem(word) for word in x]))
# vectorizing the data with maximum of 5000 features
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=10000,ngram_range=(1,3),norm='l1')
features_train = list(features_train)
features_train = tv.fit_transform(features_train).toarray()
features_test = list(features_test)
features_test = tv.fit_transform(features_test).toarray()
# getting training and testing data
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .3, random_state = 0)
output_file.write("x_train size:{}\n".format(len(features_train)))
output_file.write("x_test size:{}\n".format(len(features_test)))
print("model_1")
# Using linear support vector classifier
lsvc = LinearSVC()
# training the model
lsvc.fit(features_train, labels_train)
# getting the score of train and test data
output_file.write("\nModel: LinearSVC\n")
output_file.write("train score: {}\n".format(lsvc.score(features_train, labels_train))) 
output_file.write("test score: {}\n".format(lsvc.score(features_test, labels_test)))   
print("model_2")
# model 2:-
# Using Gaussuan Naive Bayes
gnb = GaussianNB()
gnb.fit(features_train, labels_train)
output_file.write("Model: GaussianNB\n")
output_file.write("train score: {}\n".format(gnb.score(features_train, labels_train)))  
output_file.write("test score: {}\n".format(gnb.score(features_test, labels_test)))   
print("model_3")
# model 3:-
# Logistic Regression
lr = LogisticRegression()
lr.fit(features_train, labels_train)
output_file.write("Model: LogisticRegression\n")
output_file.write("train score: {}\n".format(lr.score(features_train, labels_train)))  
output_file.write("test score: {}\n".format(lr.score(features_test, labels_test)))     
# print("model_4")
# model 4:-
# Random Forest Classifier
# rfc = RandomForestClassifier(n_estimators = 10, random_state = 0)
# rfc.fit(features_train, labels_train)
# output_file.write("Model: RandomForestClassifier\n")
# output_file.write("train score: {}\n".format(rfc.score(features_train, labels_train)))  
# output_file.write("test score: {}\n".format(rfc.score(features_test, labels_test)))    

output_file.close()