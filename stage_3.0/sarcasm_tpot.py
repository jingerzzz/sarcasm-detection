#!/usrbin/env python
#------ Featrue -------------------------#
max_features = 15000
ngram_range = (1,3)
stop_words = None
max_df = 1.0
min_df = 1
norm = 'l2'



import pandas as pd
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
from tpot import TPOTClassifier



train_df = pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
train_df = train_df.rename(columns={train_df.columns[0]:"is_sarcastic", train_df.columns[1]:"comment", train_df.columns[2]:"parent_comment"})
test_df = pd.read_csv('labeled_test.csv',delimiter=',',encoding='utf-8')
test_df = test_df.rename(columns={test_df.columns[0]:"id", test_df.columns[1]:"comment", test_df.columns[2]:"parent_comment",test_df.columns[3]:"is_sarcastic"})
train_df = train_df.dropna()
test_df = test_df.dropna()




# Relacing special symbols and digits in comment column
# re stands for Regular Expression

# train_df['comment'] = train_df['comment'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
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
ps = PorterStemmer()
features = features.apply(lambda x: x.split())
features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))

# vectorizing the data with maximum of 5000 features


tv = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words = stop_words, max_df = max_df, min_df = min_df, norm = norm)
features = list(features)
features = tv.fit_transform(features).toarray()
print(features)

# getting training and testing data
test_size = size_test/(size_test+size_train)
print(test_size)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = test_size, random_state = 0)
split_list = [features_train, features_test, labels_train, labels_test]
FNAME = "output_RandomForestClassifier.txt"
output_file = open(FNAME,'a')
    
tpot = TPOTClassifier(generations=5, population_size=30, verbosity=2, random_state=42)
tpot.fit(features_train, labels_train)
output_file.write(tpot.score(features_test, labels_test))
tpot.export('tpot_digits_pipeline.py')
output_file.close()
