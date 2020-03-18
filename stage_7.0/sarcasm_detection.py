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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from keras.models import Model, Sequential
# from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
# from keras.optimizers import RMSprop
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils.np_utils import to_categorical
# from keras.callbacks import EarlyStopping
# from keras.initializers import Constant

from pylatex import Document, LongTable, MultiColumn


def feature_extraction(X_train):
    vectorizer = CountVectorizer(ngram_range=(1, 1),max_features=10000)
    vectorizer.fit_transform(X_train)
    return vectorizer

def text_to_feature(X,vectorizer,if_tfidf=0):
    counts = vectorizer.transform(X).toarray()
    if if_tfidf==1:    
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit_transform(counts)
        return tfidf.toarray()
    else:
        return counts

def sklearn_evaluation(model_object, X_train, y_train,X_validation,y_validation,X_test, y_test):
        train_accuracy = round(model_object.score(X_train, y_train),4)
        validation_accuracy = round(model_object.score(X_validation, y_validation),4)
        test_accuracy = round(model_object.score(X_test, y_test),4)
        y_pred = model_object.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        test_precision = round(tp / (tp + fp),4)
        test_recall = round(tp / (tp + fn),4)
        test_F1 = round(2*test_precision*test_recall/(test_precision+test_recall),4)
        evaluation_dict = {"train_accuracy":train_accuracy,"validation_accuracy":validation_accuracy,"test_accuracy":test_accuracy,"test_precision":test_precision,"test_recall":test_recall,"test_F1":test_F1}
        return evaluation_dict


    
def basic_LinearSVC(X_train,y_train,X_validation,y_validation,X_test,y_test):
    model = LinearSVC()
    model.fit(X_train,y_train)
    print("LinearSVC training done.")
    evaluation_dict = sklearn_evaluation(model,X_train,y_train,X_validation,y_validation,X_test,y_test)
    return evaluation_dict

def basic_LogisticRegression(X_train,y_train,X_validation,y_validation,X_test,y_test):
    model = LogisticRegression()
    model.fit(X_train,y_train)
    print("LogisticRegression training done.")
    evaluation_dict = sklearn_evaluation(model,X_train,y_train,X_validation,y_validation,X_test,y_test)
    return evaluation_dict

def basic_GaussianNB(X_train,y_train,X_validation,y_validation,X_test,y_test):
    model = GaussianNB()
    model.fit(X_train,y_train)
    print("NaiveBayes training done.")
    evaluation_dict = sklearn_evaluation(model,X_train,y_train,X_validation,y_validation,X_test,y_test)
    return evaluation_dict

def basic_RandomForest(X_train,y_train,X_validation,y_validation,X_test,y_test):
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    print("RandomForest training done.")
    evaluation_dict = sklearn_evaluation(model,X_train,y_train,X_validation,y_validation,X_test,y_test)
    return evaluation_dict

def basic_generate_row(model_name,evaluation_dict):
    row = []
    row.append(model_name)
    row.append(evaluation_dict["train_accuracy"])
    row.append(evaluation_dict["validation_accuracy"])
    row.append(evaluation_dict["test_accuracy"])
    row.append(evaluation_dict["test_precision"])
    row.append(evaluation_dict["test_recall"])
    row.append(evaluation_dict["test_F1"])
    return row

def basic_table(X_train,y_train,X_validation,y_validation,X_test,y_test):
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)

    LinearSVC_evaluation = basic_LinearSVC(X_train,y_train,X_validation,y_validation,X_test,y_test)
    LogisticRegression_evaluation = basic_LogisticRegression(X_train,y_train,X_validation,y_validation,X_test,y_test)
    GaussianNB_evaluation = basic_GaussianNB(X_train,y_train,X_validation,y_validation,X_test,y_test)
    RandomForest_evaluation = basic_RandomForest(X_train,y_train,X_validation,y_validation,X_test,y_test)
    row_1 = basic_generate_row("LinearSVC",LinearSVC_evaluation)
    row_2 = basic_generate_row("LogisticRegression",LogisticRegression_evaluation)
    row_3 = basic_generate_row("GaussianNB",GaussianNB_evaluation)
    row_4 = basic_generate_row("RandomForest",RandomForest_evaluation)
    rows = []
    rows.append(row_1)
    rows.append(row_2)
    rows.append(row_3)
    rows.append(row_4)

    # Generate data table
    with doc.create(LongTable("l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["Model Name", "train accuracy", "validation accuracy", "test accuracy", "test precision", "test recall", "test F1"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(rows)):
                data_table.add_row(rows[i])

    doc.generate_pdf("basic_run", clean_tex=False)


def tuning_LinearSVC(X_train,y_train,X_validation,y_validation,X_test,y_test):
    param_grid = {"tol": [1e-3,1e-4,1e-5],
              "C":  [1e-5,3e-5,1e-4,3e-4,1e-3],
              "max_iter":[500,1000,2000]}
    result_list = []
    optimized_param=[0,0,0,0,0,0,0,0,0]
    for tol in param_grid["tol"]:
        for C in param_grid["C"]:
            for max_iter in param_grid["max_iter"]:
                    # try:
                        current_param_and_eval = [tol,C,max_iter]
                        model = LinearSVC(tol=tol,C=C,max_iter=max_iter)
                        model.fit(X_train,y_train)
                        evaluation_dict = sklearn_evaluation(model,X_train,y_train,X_validation,y_validation,X_test,y_test)
                        train_accuracy = evaluation_dict["train_accuracy"]
                        validation_accuracy = evaluation_dict["validation_accuracy"]
                        test_accuracy = evaluation_dict["test_accuracy"]
                        test_precision = evaluation_dict["test_precision"]
                        test_recall = evaluation_dict["test_recall"]
                        test_F1 = evaluation_dict["test_F1"]
                        current_param_and_eval.append(train_accuracy)
                        current_param_and_eval.append(validation_accuracy)
                        current_param_and_eval.append(test_accuracy)
                        current_param_and_eval.append(test_precision)
                        current_param_and_eval.append(test_recall)
                        current_param_and_eval.append(test_F1)

                        result_list.append(current_param_and_eval)
                        if current_param_and_eval[4]>optimized_param[4]:
                            optimized_param=current_param_and_eval
                    # except:
                    #     print("An exception occurs.")

    # Generate data table
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)   
    with doc.create(LongTable("l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["tol", "C", "max_iter", "training accuracy", "valid accuracy"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(result_list)):
                data_table.add_row(result_list[i][0:5])
            data_table.add_hline()
    with doc.create(LongTable("l l l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["tol", "C","max_iter", "training accuracy", "valid accuracy","test accuracy","test precision","test recall","test F1"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row(optimized_param)
            data_table.add_hline()

    doc.generate_pdf("tuning_LinearSVC", clean_tex=False)

def tuning_LogisticRegression(X_train,y_train,X_validation,y_validation,X_test,y_test):
    param_grid = {"tol": [1e-3,1e-4,1e-5],
              "C": [1e-5,3e-5,1e-4,3e-4,1e-3],
              "max_iter":[500,1000,2000]}
    result_list = []
    optimized_param=[0,0,0,0,0,0,0,0,0]
    for tol in param_grid["tol"]:
        for C in param_grid["C"]:
            for max_iter in param_grid["max_iter"]:
                    # try:
                        current_param_and_eval = [tol,C,max_iter]
                        model = LogisticRegression(tol=tol,C=C,max_iter=max_iter)
                        model.fit(X_train,y_train)
                        evaluation_dict = sklearn_evaluation(model,X_train,y_train,X_validation,y_validation,X_test,y_test)
                        train_accuracy = evaluation_dict["train_accuracy"]
                        validation_accuracy = evaluation_dict["validation_accuracy"]
                        test_accuracy = evaluation_dict["test_accuracy"]
                        test_precision = evaluation_dict["test_precision"]
                        test_recall = evaluation_dict["test_recall"]
                        test_F1 = evaluation_dict["test_F1"]
                        current_param_and_eval.append(train_accuracy)
                        current_param_and_eval.append(validation_accuracy)
                        current_param_and_eval.append(test_accuracy)
                        current_param_and_eval.append(test_precision)
                        current_param_and_eval.append(test_recall)
                        current_param_and_eval.append(test_F1)

                        result_list.append(current_param_and_eval)
                        if current_param_and_eval[4]>optimized_param[4]:
                            optimized_param=current_param_and_eval
                    # except:
                    #     print("An exception occurs.")

    # Generate data table
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)   
    with doc.create(LongTable("l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["tol", "C", "max_iter", "training accuracy", "valid accuracy"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(result_list)):
                data_table.add_row(result_list[i][0:5])
            data_table.add_hline()
    with doc.create(LongTable("l l l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["tol", "C","max_iter", "training accuracy", "valid accuracy","test accuracy","test precision","test recall","test F1"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row(optimized_param)
            data_table.add_hline()

    doc.generate_pdf("tuning_LogisticRegression", clean_tex=False)

def tuning_GaussianNB(X_train,y_train,X_validation,y_validation,X_test,y_test):
    param_grid = {"var_smoothing": [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],}
    result_list = []
    optimized_param=[0,0,0,0,0,0,0]
    for var_smoothing  in param_grid["var_smoothing"]:
                    # try:
                        current_param_and_eval = [var_smoothing]
                        model = GaussianNB(var_smoothing=var_smoothing)
                        model.fit(X_train,y_train)
                        evaluation_dict = sklearn_evaluation(model,X_train,y_train,X_validation,y_validation,X_test,y_test)
                        train_accuracy = evaluation_dict["train_accuracy"]
                        validation_accuracy = evaluation_dict["validation_accuracy"]
                        test_accuracy = evaluation_dict["test_accuracy"]
                        test_precision = evaluation_dict["test_precision"]
                        test_recall = evaluation_dict["test_recall"]
                        test_F1 = evaluation_dict["test_F1"]
                        current_param_and_eval.append(train_accuracy)
                        current_param_and_eval.append(validation_accuracy)
                        current_param_and_eval.append(test_accuracy)
                        current_param_and_eval.append(test_precision)
                        current_param_and_eval.append(test_recall)
                        current_param_and_eval.append(test_F1)

                        result_list.append(current_param_and_eval)
                        if current_param_and_eval[2]>optimized_param[2]:
                            optimized_param=current_param_and_eval
                    # except:
                    #     print("An exception occurs.")

    # Generate data table
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)   
    with doc.create(LongTable("l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["var_smoothing", "training accuracy", "valid accuracy"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(result_list)):
                data_table.add_row(result_list[i][0:3])
            data_table.add_hline()
    with doc.create(LongTable("l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["var_smoothing", "training accuracy", "valid accuracy","test accuracy","test precision","test recall","test F1"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row(optimized_param)
            data_table.add_hline()

    doc.generate_pdf("tuning_GaussianNB", clean_tex=False)

def tuning_RandomForest(X_train,y_train,X_validation,y_validation,X_test,y_test):
    param_grid = {"max_features":["auto","sqrt","log2"],
              "n_estimators": [50,100,200],
              "min_sample_leaf":[25,50,100],
              "max_depth": [None,10,20,40,80],
              }
    result_list = []
    optimized_param=[0,0,0,0,0,0,0,0,0,0]
    for max_features in param_grid["max_features"]:
        for n_estimators in param_grid["n_estimators"]:
            for min_sample_leaf in param_grid["min_sample_leaf"]:
                    for max_depth in param_grid["max_depth"]:
                    # try:
                        current_param_and_eval = [max_features,n_estimators,min_sample_leaf,max_depth]
                        model = RandomForestClassifier(max_features=max_features,n_estimators=n_estimators,min_samples_leaf=min_sample_leaf,max_depth=max_depth)
                        model.fit(X_train,y_train)
                        evaluation_dict = sklearn_evaluation(model,X_train,y_train,X_validation,y_validation,X_test,y_test)
                        train_accuracy = evaluation_dict["train_accuracy"]
                        validation_accuracy = evaluation_dict["validation_accuracy"]
                        test_accuracy = evaluation_dict["test_accuracy"]
                        test_precision = evaluation_dict["test_precision"]
                        test_recall = evaluation_dict["test_recall"]
                        test_F1 = evaluation_dict["test_F1"]
                        current_param_and_eval.append(train_accuracy)
                        current_param_and_eval.append(validation_accuracy)
                        current_param_and_eval.append(test_accuracy)
                        current_param_and_eval.append(test_precision)
                        current_param_and_eval.append(test_recall)
                        current_param_and_eval.append(test_F1)

                        result_list.append(current_param_and_eval)
                        if current_param_and_eval[5]>optimized_param[5]:
                            optimized_param=current_param_and_eval
                    # except:
                    #     print("An exception occurs.")

    # Generate data table
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)   
    with doc.create(LongTable("l l l l l l ")) as data_table:
            data_table.add_hline()
            data_table.add_row(["max_features","n_estimators","min_sample_leaf","max_depth", "training accuracy", "valid accuracy"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(result_list)):
                data_table.add_row(result_list[i][0:6])
            data_table.add_hline()
    with doc.create(LongTable("l l l l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["max_features","n_estimators","min_sample_leaf","max_depth", "training accuracy", "valid accuracy","test accuracy","test precision","test recall","test F1"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row(optimized_param)
            data_table.add_hline()

    doc.generate_pdf("tuning_RandomForest", clean_tex=False)


## load data
f = open('split_data\X_train.txt', 'r')
X_train = f.read().splitlines()
f.close()

f = open('split_data\y_train.txt', 'r')
y_train = f.read().splitlines()
for i in range(len(y_train)):
    y_train[i]=int(y_train[i])
f.close()

f = open('split_data\X_test.txt', 'r')
X_test = f.read().splitlines()
f.close()

f = open('split_data\y_test.txt', 'r')
y_test = f.read().splitlines()
for i in range(len(y_test)):
    y_test[i]=int(y_test[i])
f.close()

f = open('split_data\X_validation.txt', 'r')
X_validation = f.read().splitlines()
f.close()

f = open('split_data\y_validation.txt', 'r')
y_validation = f.read().splitlines()
for i in range(len(y_validation)):
    y_validation[i]=int(y_validation[i])
f.close()

# X_train = X_train[0:200]
# y_train = y_train[0:200]
# X_validation = X_validation[0:50]
# y_validation = y_validation[0:50]
# X_test = X_test[0:80]
# y_test = y_test[0:80]



## feature extraction
vectorizer=feature_extraction(X_train)
X_vec_train = text_to_feature(X_train,vectorizer,if_tfidf=0)
X_vec_validation = text_to_feature(X_validation,vectorizer,if_tfidf=0)
X_vec_test = text_to_feature(X_test,vectorizer,if_tfidf=0)
print("feature transform done.")

## model training and prediction
# tuning_LinearSVC(X_vec_train,y_train,X_vec_validation,y_validation,X_vec_test,y_test)
# tuning_LogisticRegression(X_vec_train,y_train,X_vec_validation,y_validation,X_vec_test,y_test)
# tuning_GaussianNB(X_vec_train,y_train,X_vec_validation,y_validation,X_vec_test,y_test)
# tuning_RandomForest(X_vec_train,y_train,X_vec_validation,y_validation,X_vec_test,y_test)