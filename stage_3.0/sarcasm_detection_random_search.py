################ Parameters selection area #################
#------ Model -------------------------#
model_test ={"LinearSVC":0,"GaussianNB":0,"LogisticRegression":0,"RandomForestClassifier":1,"SVC":0}
#------ Featrue -------------------------#
max_features = 15000
ngram_range = (1,3)
stop_words = None
max_df = 1.0
min_df = 1
norm = 'l2'

#-----LinearSVC--------------------------#
lsvc_penalty = 'l2'
lsvc_c =1.0
lsvc_tol = 0.0001
lsvc_dual = False


#-----GaussianNB--------------------------#
gnb_var_smoothing = 1e-9


#-----LogisticRegression--------------------------#
lr_penalty = 'l2'
lr_tol = 0.0001
lr_c = 1.0


#-----RandomForestClassifier--------------------------#
rfc_n_estimators = 'warn'
rfc_oob_score = False
rfc_max_depth = 10

#-----SVC--------------------------#
svc_c = 1.0
svc_kernel = 'rbf'
svc_tol =0.001

##########################################



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
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

from sarcasm_detection_utilities import *

def feature_extraction():
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
    return split_list
    
def run_models(features_train, features_test, labels_train, labels_test):
    feature_extract_params_dict = {"max_features":max_features,"ngram_range":ngram_range,"stop_words":stop_words,"max_df":max_df,"min_df":min_df}

    LinearSVC_hyper_params_dict = {"lsvc_penalty":lsvc_penalty,"lsvc_c":lsvc_c,"lsvc_tol":lsvc_tol}
    GaussianNB_hyper_params_dict = {"gnb_var_smoothing":gnb_var_smoothing}
    LogisticRegression_hyper_params_dict = {"lr_penalty":lr_penalty,"lr_tol":lr_tol,"lr_c":lr_c}
    RandomForestClassifier_hyper_params_dict = {"rfc_n_estimators":rfc_n_estimators,"rfc_oob_score":rfc_oob_score,"rfc_max_depth":rfc_max_depth}
    SVC_hyper_params_dict = {"svc_c":svc_c,"svc_kernel":svc_kernel,"svc_tol":svc_tol}



    if model_test["LinearSVC"]==1:
        record_parameters("LinearSVC",feature_extract_params_dict ,LinearSVC_hyper_params_dict ,len(features_train) ,len(features_test))
        lsvc = LinearSVC(penalty=lsvc_penalty,C=lsvc_c, tol=lsvc_tol, dual=lsvc_dual)
        lsvc.fit(features_train, labels_train)
        scores_dict = calculate_scores(lsvc,features_train,features_test,labels_train,labels_test)
        record_results("LinearSVC",scores_dict)

    if model_test['GaussianNB']==1: 
        record_parameters("GaussianNB",feature_extract_params_dict ,GaussianNB_hyper_params_dict ,len(features_train) ,len(features_test))
        gnb = GaussianNB(var_smoothing=gnb_var_smoothing)
        gnb.fit(features_train, labels_train)
        scores_dict = calculate_scores(gnb,features_train,features_test,labels_train,labels_test)
        record_results("GaussianNB",scores_dict)

    if model_test["LogisticRegression"]==1:      
        record_parameters("LogisticRegression",feature_extract_params_dict ,LogisticRegression_hyper_params_dict ,len(features_train) ,len(features_test))
        lr = LogisticRegression(C=lr_c,penalty=lr_penalty,tol=lr_tol)
        lr.fit(features_train, labels_train)
        scores_dict = calculate_scores(lr,features_train,features_test,labels_train,labels_test)
        record_results("LogisticRegression",scores_dict)

    if model_test["RandomForestClassifier"]==1:  
        record_parameters("RandomForestClassifier",feature_extract_params_dict ,RandomForestClassifier_hyper_params_dict ,len(features_train) ,len(features_test))        
        rfc = RandomForestClassifier(n_estimators = rfc_n_estimators,oob_score=rfc_oob_score,max_depth=rfc_max_depth)
        rfc.fit(features_train, labels_train)
        scores_dict = calculate_scores(rfc,features_train,features_test,labels_train,labels_test)
        record_results("RandomForestClassifier",scores_dict)
    
    if model_test["SVC"]==1:  
        record_parameters("SVC",feature_extract_params_dict ,SVC_hyper_params_dict ,len(features_train) ,len(features_test))      
        svc = SVC(kernel=svc_kernel,C=svc_c,tol=svc_tol)
        svc.fit(features_train, labels_train)
        scores_dict = calculate_scores(svc,features_train,features_test,labels_train,labels_test)
        record_results("SVC",scores_dict)

def run_model_1(features_train, features_test, labels_train, labels_test):


    features = pd.concat([features_train, features_test],ignore_index=True)
    labels = pd.concat([labels_train, labels_test],ignore_index=True)
    clf = RandomForestClassifier(n_estimators=20)
    param_dist = {"max_depth": [3, None],                 #list
              "max_features": sp_randint(1, 11),          #distribution
              "min_samples_split": sp_randint(2, 11),     #distribution
              "bootstrap": [True, False],                 #list
              "criterion": ["gini", "entropy"]}           #list
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                    n_iter=n_iter_search, cv=5, iid=False)
    random_search.fit(features, labels)

    print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
    means = random_search.cv_results_['mean_test_score']
    stds = random_search.cv_results_['std_test_score']
    params = random_search.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


    


    
if __name__ == "__main__":
    split_list = feature_extraction()
        
    run_model_1(split_list[0],split_list[1],split_list[2],split_list[3])
    
        
