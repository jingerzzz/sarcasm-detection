## v4: Add stemming

import pandas as pd
import numpy as np
import re
import random
from io import StringIO
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score




train_df = pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
# test_df = pd.read_csv('test.tsv',delimiter='\t',encoding='utf-8')

train_df = train_df.rename(columns={train_df.columns[0]:"category", train_df.columns[1]:"comment", train_df.columns[2]:"parent_comment"})
# test_df = test_df.rename(columns={test_df.columns[0]:"category", test_df.columns[1]:"comment", test_df.columns[2]:"parent_comment"})

# Rename the column names for later use
df = train_df[10001:30000]
data_size = len(df)


FNAME = "output_v4_data_size={}.txt".format(data_size)
output_file = open(FNAME,'w')
output_file.write("Dataset size:{}\n".format(len(df)))

tfidf = TfidfVectorizer(max_features = 5000)


df['comment'] = df['comment'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
features = df['comment']

ps = PorterStemmer()
features = features.apply(lambda x: x.split())
features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))
features = tfidf.fit_transform(df.comment).toarray()
features = list(features)
# output_file.write("features shape:{}\n".format(features.shape))
print(features)

labels = df.category

print("done_1")


x_train, x_test, y_train, y_test = train_test_split(features, labels,test_size = .05, random_state = 0)

output_file.write("x_train size:{}\n".format(len(x_train)))
output_file.write("x_test size:{}\n".format(len(x_test)))

# Divide dataset into training set and testing set


# Using linear support vector classifier
lsvc = LinearSVC()
# training the model
lsvc.fit(x_train, y_train)  
# getting the score of train and test data
output_file.write("train score: {}\n".format(lsvc.score(x_train, y_train)))
output_file.write("test score: {}\n".format(lsvc.score(x_test, y_test)))






output_file.close()