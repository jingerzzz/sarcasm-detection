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
import csv

test_df = pd.read_csv('test.tsv',delimiter='\t',encoding='utf-8')
label_df = pd.read_csv('truth.csv',delimiter=',',encoding='utf-8')

test_df = test_df.join(label_df.set_index('id')[['label']], on='id')
test_df.to_csv ('labeled_test.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
