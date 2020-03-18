import pandas as pd, numpy as np, re
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
train_df = train_df.rename(columns={train_df.columns[0]:"is_sarcastic", train_df.columns[1]:"comment", train_df.columns[2]:"parent_comment"})
test_df = pd.read_csv('labeled_test.csv',delimiter=',',encoding='utf-8')
test_df = test_df.rename(columns={test_df.columns[0]:"id", test_df.columns[1]:"comment", test_df.columns[2]:"parent_comment",test_df.columns[3]:"is_sarcastic"})
train_df = train_df.dropna()
test_df = test_df.dropna()

features_train = train_df['comment'][0:10]
features_test = test_df['comment'][0:10]

features =pd.concat( [features_train, features_test],ignore_index=True)
print(features)
print(features[15])

features_train = train_df['comment'][0:10]
features_test = train_df['comment'][10:20]
print(features_train)
print(features_test)


# features_test = test_df['comment']
# exception_list=[]
# for i in range(len(features_test)):
#     features = features_test[i:i+1]
#     # print(features_test)
#     try:
#         features.apply(lambda x: x.split())

#     except:
#         print(features_test[i:i+1])
#         exception_list.append(i)

# print(exception_list)
