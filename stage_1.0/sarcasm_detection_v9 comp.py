## v9_comp: Use doc2vec to extract feature instead of BOW, Tfidf based on the dataset containing 'headline'
import multiprocessing

from tqdm import tqdm
from sklearn import utils
import pandas as pd, numpy as np, re
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Loading data from json file
# Loading data from json file
data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines = True)
# Relacing special symbols and digits in headline column
# re stands for Regular Expression
data['headline'] = data['headline'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))


FNAME = "output_v9_comp_data_size={}.txt".format(len(data))
output_file = open(FNAME,'w')
output_file.write("Dataset size:{}\n".format(len(data)))

# Relacing special symbols and digits in comment column
# re stands for Regular Expression



train, test = train_test_split(data,test_size = 0.3, random_state=42)

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

train_tagged = train.apply(lambda r: TaggedDocument(words=tokenize_text(r['headline']), tags=[r.is_sarcastic]), axis=1)
test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['headline']), tags=[r.is_sarcastic]), axis=1)



cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])


for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
output_file.write('Testing accuracy %s\n' % accuracy_score(y_test, y_pred))
output_file.write('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))