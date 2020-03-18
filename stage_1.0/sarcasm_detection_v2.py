## v2: Change TfidfVectorizer instance

import pandas as pd
import numpy as np
import random
from io import StringIO
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
df = train_df
data_size = len(df)


FNAME = "output_v2_data_size={}.txt".format(data_size)
output_file = open(FNAME,'w')
output_file.write("Dataset size:{}\n".format(len(df)))


tfidf = TfidfVectorizer(max_features = 5000)




features = tfidf.fit_transform(df.comment).toarray()
output_file.write("features shape:{}\n".format(features.shape))
print(features)
# f_feature = open("feature.txt",'w')
# for i in range(len(features)):
#     f_feature.write("{}".format(features[i]))
# f_feature.close()
labels = df.category

print("done_1")


x_train, x_test, y_train, y_test = train_test_split(features, labels,test_size = .25, random_state = 0)

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



# count_vect = CountVectorizer()
# x_train_counts = count_vect.fit_transform(x_train)
# tfidf_transformer = TfidfTransformer()
# x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
# print(x_train_tfidf)
# print("done_2")

# clf = MultinomialNB().fit(x_train_tfidf, y_train)
print("done_3")
# print(clf.predict(count_vect.transform(["Thanks Obama!"])))
# models = [
#     RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
#     LinearSVC(),
#     MultinomialNB(),
#     LogisticRegression(random_state=0),
# ]
# # Test the accuracies of different models
# CV = 5
# cv_df = pd.DataFrame(index=range(CV * len(models)))
# entries = []
# index=1
# for model in models:
#     model_name = model.__class__.__name__
#     accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
#     for fold_index, accuracy in enumerate(accuracies):
#         entries.append((model_name, fold_index, accuracy))
#     print("model_{}".format(index))
#     index=index+1
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_index', 'accuracy'])
# cv_df_mean = cv_df.groupby('model_name').accuracy.mean()
# output_file.write("\nAccuracy\n")
# for name, value in cv_df_mean.items():
#     output_file.write("{}  {}\n".format(name,value))



output_file.close()