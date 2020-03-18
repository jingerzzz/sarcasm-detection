import pandas as pd
import numpy as np
import re
import random
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.initializers import Constant

train_df = pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
train_df = train_df.rename(columns={train_df.columns[0]:"is_sarcastic", train_df.columns[1]:"comment", train_df.columns[2]:"parent_comment"})
test_df = pd.read_csv('labeled_test.csv',delimiter=',',encoding='utf-8')
test_df = test_df.rename(columns={test_df.columns[0]:"id", test_df.columns[1]:"comment", test_df.columns[2]:"parent_comment",test_df.columns[3]:"is_sarcastic"})
train_df = train_df.dropna()
test_df = test_df.dropna()




# Relacing special symbols and digits in comment column
# re stands for Regular Expression

train_df['comment'] = train_df['comment'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
# test_df['comment'] = test_df['comment'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
# getting features and labels

features_train = train_df['comment']
features_test = test_df['comment']
size_train = len(features_train)
size_test = len(features_test)
features = pd.concat([features_train, features_test],ignore_index=True)


labels_train = train_df['is_sarcastic']
labels_test = test_df['is_sarcastic']
labels = pd.concat([labels_train, labels_test],ignore_index=True)
# Stemming our data
features= features.apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))


ps = PorterStemmer()
features = features.apply(lambda x: x.split())
features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))



# getting training and testing data
test_size_ratio = size_test/(size_test+size_train)




tokenizer = Tokenizer()
tokenizer.fit_on_texts(features)
labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = test_size_ratio, random_state = 0)
features_train = features_train.to_list()
features_test = features_test.to_list()
labels_train = labels_train.tolist()
labels_test = labels_test.tolist()

## split validation set from train set
# print(isinstance(features_train,list))
# print(labels_train)
random.seed(1)
total_size = len(features_train)
validation_size = int(0.2*total_size)
print("total_size:{}, test_size:{}, validation_size:{}".format(total_size,total_size-validation_size,validation_size))
total_list = list(range(0,total_size))
validation_index_list = random.sample(total_list,validation_size)
# print(validation_index_list)
features_validation = [features_train[x] for x in validation_index_list]
X_train = [features_train[x] for x in range(0,total_size) if x not in validation_index_list]
labels_validation = [labels_train[x] for x in validation_index_list]
Y_train = [labels_train[x] for x in range(0,total_size) if x not in validation_index_list]
print("train size:{}, validation size:{}, test size:{}".format(len(X_train),len(features_validation),len(features_test)))
X_validation = features_validation
Y_validation = labels_validation


