import pandas as pd, numpy as np, re
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

def record_parameters(model_name,feature_extract_params_dict, hyper_params_dict,size_train,size_test):
    FNAME = "output_{}.txt".format(model_name)
    output_file = open(FNAME,'a')
    output_file.write("----------------------------------------------\n")

    output_file.write("Parameters:\n\tFeature Extraction(tfidf):")
    for params_name, params_value in feature_extract_params_dict.items():
        output_file.write("\n\t\t{}: {}".format(params_name,params_value))
    output_file.write("\n")

    output_file.write("\tModel:linearSVC:")
    for params_name, params_value in hyper_params_dict.items():
        output_file.write("\n\t\t{}: {}".format(params_name,params_value))
    output_file.write("\n")
    output_file.write("\nData size:\n")
    output_file.write("\ttotal size: {}\n".format(size_train+size_test))
    output_file.write("\ttrain size: {}\n".format(size_train))
    output_file.write("\ttest size: {}\n".format(size_test))
    output_file.close()

def record_results(model_name,scores_dict):
    # getting the score of train and test data
    FNAME = "output_{}.txt".format(model_name)
    output_file = open(FNAME,'a')
    
    output_file.write("\nResults:\n")
    for score_name, score_value in scores_dict.items():
        output_file.write("\t\t{}: {}\n".format(score_name, score_value))    
    output_file.write("----------------------------------------------\n")
    print("Training of {} is finished.".format(model_name))
    output_file.close()

def calculate_scores(model_object, features_train, features_test, labels_train, labels_test):
        train_score = model_object.score(features_train, labels_train)
        test_score = model_object.score(features_test, labels_test)
        predicted_labels = model_object.predict(features_test)
        tn, fp, fn, tp = confusion_matrix(labels_test, predicted_labels).ravel()
        precision_score = tp / (tp + fp)
        recall_score = tp / (tp + fn)
        F1_score = 2*precision_score*recall_score/(precision_score+recall_score)
        scores_dict = {"train_score":train_score,"test_score":test_score,"precision_score":precision_score,"recall_score":recall_score,"F1_score":F1_score}
        return scores_dict