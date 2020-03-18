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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

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
X_train = train_df['comment']
X_test = test_df['comment']
size_train = len(X_train)
size_test = len(X_test)



y_train = train_df['is_sarcastic']
y_test = test_df['is_sarcastic']




# getting training and testing data
test_size_ratio = size_test/(size_test+size_train)




# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X_train)
labels = to_categorical(y_train)
X_train = X_train.to_list()
X_test = X_test.to_list()
y_train = y_train.tolist()
y_test = y_test.tolist()

for i in range(len(X_train)):
    if "\n" in X_train[i]:
        X_train [i] = X_train[i].replace("\n"," ")
for i in range(len(X_test)):
    if "\n" in X_test[i]:
        X_test[i] = X_test[i].replace("\n"," ")
## split validation set from train set
random.seed(1)
total_size = len(X_train)
validation_size = int(0.2*total_size)
print("total_size:{}, train_size:{}, validation_size:{}".format(total_size,total_size-validation_size,validation_size))
total_list = list(range(0,total_size))
validation_index_list = random.sample(total_list,validation_size)
# print(validation_index_list)
X_validation = [X_train[x] for x in validation_index_list]
X_train = [X_train[x] for x in range(0,total_size) if x not in validation_index_list]
Y_validation = [y_train[x] for x in validation_index_list]
Y_train = [y_train[x] for x in range(0,total_size) if x not in validation_index_list]
print("train size:{}, validation size:{}, test size:{}".format(len(X_train),len(X_validation),len(X_test)))
X_validation = X_validation
Y_test = y_test


with open("split_data\X_train.txt", 'w') as output:
    for row in X_train:
        output.write(str(row) + '\n')

with open("split_data\X_test.txt", 'w') as output:
    for row in X_test:
        output.write(str(row) + '\n')

with open("split_data\X_validation.txt", 'w') as output:
    for row in X_validation:
        output.write(str(row) + '\n')

with open("split_data\y_train.txt", 'w') as output:
    for row in Y_train:
        output.write(str(row) + '\n')

with open("split_data\y_test.txt", 'w') as output:
    for row in Y_test:
        output.write(str(row) + '\n')

with open("split_data\y_validation.txt", 'w') as output:
    for row in Y_validation:
        output.write(str(row) + '\n')


