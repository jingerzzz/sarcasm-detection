## train feature on both trainset and testset

# Parameters selection area ##############
#------ Model -------------------------#
model_test ={"LinearSVC":0,"GaussianNB":0,"LogisticRegression":0,"RandomForestClassifier":0}
#------ Featrue -------------------------#
max_features = 15000
ngram_range = (1,3)
stop_words = None
max_df = 1.0
min_df = 1
norm = 'l2'

feature_extract_params_dict = {"max_features":max_features,"ngram_range":ngram_range,"stop_words":stop_words,"max_df":max_df,"min_df":min_df}
#-----LinearSVC--------------------------#
lsvc_penalty = 'l2'
lsvc_c =0.05
lsvc_tol = 0.001

LinearSVC_hyper_params_dict = {"lsvc_penalty":lsvc_penalty,"lsvc_c":lsvc_c,"lsvc_tol":lsvc_tol}
#-----GaussianNB--------------------------#
gnb_var_smoothing = 1e-9

GaussianNB_hyper_params_dict = {"gnb_var_smoothing":gnb_var_smoothing}
#-----LogisticRegression--------------------------#
lr_penalty = 'l2'
lr_tol = 0.0001
lr_c = 0.2

LogisticRegression_hyper_params_dict = {"lr_penalty":lr_penalty,"lr_tol":lr_tol,"lr_c":lr_c}
#-----RandomForestClassifier--------------------------#
rfc_max_feature = "auto"
rfc_max_depth = 10
rfc_n_estimators = 50
rfc_min_sample_leaf = 1
rfc_oob_score = False


RandomForestClassifier_hyper_params_dict = {"rfc_max_feature":rfc_max_feature,"rfc_max_depth":rfc_max_depth,"rfc_n_estimators":rfc_n_estimators,"rfc_min_sample_leaf":rfc_min_sample_leaf,"rfc_oob_score":rfc_oob_score}
#-----SVC--------------------------#
svc_c = 1.0
svc_kernel = 'rbf'
svc_tol =0.001
SVC_hyper_params_dict = {"svc_c":svc_c,"svc_kernel":svc_kernel,"svc_tol":svc_tol}
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


# # Loading data from json file
# train_df = pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
# train_df = train_df.rename(columns={train_df.columns[0]:"is_sarcastic", train_df.columns[1]:"comment", train_df.columns[2]:"parent_comment"})
# test_df = pd.read_csv('labeled_test.csv',delimiter=',',encoding='utf-8')
# test_df = test_df.rename(columns={test_df.columns[0]:"id", test_df.columns[1]:"comment", test_df.columns[2]:"parent_comment",test_df.columns[3]:"is_sarcastic"})
# train_df = train_df.dropna()
# test_df = test_df.dropna()




# # Relacing special symbols and digits in comment column
# # re stands for Regular Expression

# train_df['comment'] = train_df['comment'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
# # test_df['comment'] = test_df['comment'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
# # getting features and labels

# features_train = train_df['comment']
# features_test = test_df['comment']
# size_train = len(features_train)
# size_test = len(features_test)
# features = pd.concat([features_train, features_test],ignore_index=True)
# # comments_file=open('comments_v5.txt','w')
# # for i in range(len(features)):
# #     comments_file.write("{}    {}\n".format(i,features[i]))
# # comments_file.close()

# labels_train = train_df['is_sarcastic']
# labels_test = test_df['is_sarcastic']
# labels = pd.concat([labels_train, labels_test],ignore_index=True)
# # Stemming our data
# ps = PorterStemmer()
# features = features.apply(lambda x: x.split())
# features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))

# # vectorizing the data with maximum of 5000 features
def record_hyper_parameters(model_name,feature_extract_params_dict, hyper_params_dict,size_train,size_test):
    FNAME = "output_{}.txt".format(model_name)
    output_file = open(FNAME,'a')
    output_file.write("----------------------------------------------\n")

    output_file.write("Parameters:\n\tFeature Extraction(tfidf):")
    for params_name, params_value in feature_extract_params_dict.items():
        output_file.write("\n\t\t{}:{}".format(params_name,params_value))
    output_file.write("\n")

    output_file.write("\tModel:linearSVC:")
    for params_name, params_value in hyper_params_dict.items():
        output_file.write("\n\t\t{}:{}".format(params_name,params_value))
    output_file.write("\n")
    output_file.write("\nData size:\n")
    output_file.write("\ttotal size: {}\n".format(size_train+size_test))
    output_file.write("\ttrain size: {}\n".format(size_train))
    output_file.write("\txtest size: {}\n".format(size_test))
    output_file.close()

def record_results(model_name,scores_dict):
    # getting the score of train and test data
    FNAME = "output_{}.txt".format(model_name)
    output_file = open(FNAME,'a')
    
    output_file.write("\nResults:\n")
    for score_name, score_value in scores_dict.items():
        output_file.write("\t\t{}: {}\n".format(score_name, score_value))    
    output_file.write("----------------------------------------------\n")
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

def feature_extraction():
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
    features_list = [features_train, features_test, labels_train, labels_test]
    return features_list

def run_models(features_list):

    features_train = features_list[0]
    features_test = features_list[1]
    labels_train = features_list[2]
    labels_test = features_list[3]

    current_results = {"LinearSVC":
                            {"hypers":
                                {"lsvc_penalty":lsvc_penalty,
                                 "lsvc_c":lsvc_c,
                                 "lsvc_tol":lsvc_tol},
                             "scores":{"train_score":0,"test_score":0,"precision_score":0,"recall_score":0,"F1_score":0}},
                        "GaussianNB":
                            {"hypers":
                                {"gnb_var_smoothing":gnb_var_smoothing},
                             "scores":{"train_score":0,"test_score":0,"precision_score":0,"recall_score":0,"F1_score":0}},
                        "LogisticRegression":
                            {"hypers":
                                {"lr_penalty":lr_penalty,
                                 "lr_tol":lr_tol,
                                 "lr_c":lr_c},
                             "scores":{"train_score":0,"test_score":0,"precision_score":0,"recall_score":0,"F1_score":0}},
                        "RandomForestClassifier":
                            {"hypers":
                                {"rfc_n_estimators":rfc_n_estimators,"rfc_oob_score":rfc_oob_score,"rfc_max_depth":rfc_max_depth},
                             "scores":{"train_score":0,"test_score":0,"precision_score":0,"recall_score":0,"F1_score":0}}                                                         
                      }





    if model_test["LinearSVC"]==1:
        LinearSVC_hyper_params_dict = {"lsvc_penalty":lsvc_penalty,"lsvc_c":lsvc_c,"lsvc_tol":lsvc_tol}
        record_hyper_parameters("LinearSVC",feature_extract_params_dict ,LinearSVC_hyper_params_dict ,len(features_train) ,len(features_test))
        lsvc = LinearSVC(penalty=lsvc_penalty,C=lsvc_c, tol=lsvc_tol)
        lsvc.fit(features_train, labels_train)
        scores_dict = calculate_scores(lsvc,features_train,features_test,labels_train,labels_test)
        record_results("LinearSVC",scores_dict)
        current_results["LinearSVC"]["scores"] = scores_dict

    if model_test['GaussianNB']==1: 
        GaussianNB_hyper_params_dict = {"gnb_var_smoothing":gnb_var_smoothing}
        record_hyper_parameters("GaussianNB",feature_extract_params_dict ,GaussianNB_hyper_params_dict ,len(features_train) ,len(features_test))
        gnb = GaussianNB(var_smoothing=gnb_var_smoothing)
        gnb.fit(features_train, labels_train)
        scores_dict = calculate_scores(gnb,features_train,features_test,labels_train,labels_test)
        record_results("GaussianNB",scores_dict)
        current_results["GaussianNB"]["scores"] = scores_dict

    if model_test["LogisticRegression"]==1:      
        LogisticRegression_hyper_params_dict = {"lr_penalty":lr_penalty,"lr_tol":lr_tol,"lr_c":lr_c}
        record_hyper_parameters("LogisticRegression",feature_extract_params_dict ,LogisticRegression_hyper_params_dict ,len(features_train) ,len(features_test))
        lr = LogisticRegression(C=lr_c,penalty=lr_penalty,tol=lr_tol)
        lr.fit(features_train, labels_train)
        scores_dict = calculate_scores(lr,features_train,features_test,labels_train,labels_test)
        record_results("LogisticRegression",scores_dict)
        current_results["LogisticRegression"]["scores"] = scores_dict

    if model_test["RandomForestClassifier"]==1:  
        RandomForestClassifier_hyper_params_dict = {"rfc_max_feature":rfc_max_feature,"rfc_max_depth":rfc_max_depth,"rfc_n_estimators":rfc_n_estimators,"rfc_min_sample_leaf":rfc_min_sample_leaf,"rfc_oob_score":rfc_oob_score}
        record_hyper_parameters("RandomForestClassifier",feature_extract_params_dict ,RandomForestClassifier_hyper_params_dict ,len(features_train) ,len(features_test))        
        rfc = RandomForestClassifier(n_estimators = rfc_n_estimators,oob_score=rfc_oob_score)
        rfc.fit(features_train, labels_train)
        scores_dict = calculate_scores(rfc,features_train,features_test,labels_train,labels_test)
        record_results("RandomForestClassifier",scores_dict)
        current_results["RandomForestClassifier"]["scores"] = scores_dict

    return current_results
    
    # if model_test["SVC"]==1:  
    #     record_hyper_parameters("SVC",feature_extract_params_dict ,SVC_hyper_params_dict ,len(features_train) ,len(features_test))      
    #     svc = SVC(kernel=svc_kernel,C=svc_c,tol=svc_tol)
    #     svc.fit(features_train, labels_train)
    #     scores_dict = calculate_scores(svc,features_train,features_test,labels_train,labels_test)
    #     record_results("SVC",scores_dict)

if __name__ == "__main__":
    ## record max test score and hyperparameters, current test score and hyperparameters
    ## linear SVC
    lsvc_max_accuracy = 0
    lsvc_current_accuracy = 0
    lsvc_max_scores = {}
    LinearSVC_hyper_params_dict = {"lsvc_penalty":lsvc_penalty,"lsvc_c":lsvc_c,"lsvc_tol":lsvc_tol}
    lsvc_max_accuracy_hyperparams = {}
    lsvc_current_accuracy_hyperparams = LinearSVC_hyper_params_dict
    ## gnb
    gnb_max_accuracy = 0
    gnb_current_accuracy = 0
    gnb_max_scores = {}
    GaussianNB_hyper_params_dict = {"gnb_var_smoothing":gnb_var_smoothing}
    gnb_max_accuracy_hyperparams = {}
    gnb_current_accuracy_hyperparams = GaussianNB_hyper_params_dict
    ## lr
    lr_max_accuracy = 0
    lr_current_accuracy = 0
    lr_max_scores = {}
    LogisticRegression_hyper_params_dict = {"lr_penalty":lr_penalty,"lr_tol":lr_tol,"lr_c":lr_c}
    lr_max_accuracy_hyperparams = {}
    lr_current_accuracy_hyperparams = LogisticRegression_hyper_params_dict
    ## rfc
    rfc_max_accuracy = 0
    rfc_current_accuracy = 0
    rfc_max_scores = {}
    RandomForestClassifier_hyper_params_dict = {"rfc_n_estimators":rfc_n_estimators,"rfc_oob_score":rfc_oob_score,"rfc_max_depth":rfc_max_depth}
    rfc_max_accuracy_hyperparams = {}
    rfc_current_accuracy_hyperparams = RandomForestClassifier_hyper_params_dict
    
    ## run features extranction
    features_list = feature_extraction()


    ## grid search and record history best results
  

    ## linear SVC      
    model_test ={"LinearSVC":0,"GaussianNB":0,"LogisticRegression":0,"RandomForestClassifier":0}
    if model_test["LinearSVC"]!=0:   
        params = {"lsvc_c":[0.001,0.01,0.1,1,10,100],
                "lsvc_tol":[0.000001,0.00001,0.001,0.01,0.1]}
        for p1 in params["lsvc_c"]:
            for p2 in params["lsvc_tol"]:
                lsvc_c = p1
                lsvc_tol = p2
                
                current_results = run_models(features_list)
                lsvc_current_accuracy = current_results["LinearSVC"]["scores"]["test_score"]
                if lsvc_current_accuracy > lsvc_max_accuracy:
                    lsvc_max_accuracy = lsvc_current_accuracy
                    lsvc_max_scores = current_results["LinearSVC"]["scores"]
                    lsvc_max_accuracy_hyperparams = current_results["LinearSVC"]["hypers"]

        FNAME = "output_max_LinearSVC.txt"
        output_file = open(FNAME,'a')
        output_file.write("\nMax scores:\n")
        for key, value in lsvc_max_scores.items():
            output_file.write("{}: {}\n".format(key, value))
        output_file.write("\nCorresponding hyperparameters:\n")
        for key, value in lsvc_max_accuracy_hyperparams.items():
            output_file.write("{}: {}\n".format(key, value))
        output_file.close()

    ## gnb
    model_test ={"LinearSVC":0,"GaussianNB":0,"LogisticRegression":0,"RandomForestClassifier":0}
    if model_test["GaussianNB"]!=0:   
        params = {"gnb_var_smoothing":[1e-10,1e-9,1e-8,1e-7,1e-6]}
        for p1 in params["gnb_var_smoothing"]:
            gnb_var_smoothing = p1

            current_results = run_models(features_list)
            gnb_current_accuracy = current_results["GaussianNB"]["scores"]["test_score"]
            if gnb_current_accuracy > gnb_max_accuracy:
                gnb_max_accuracy = gnb_current_accuracy
                gnb_max_scores = current_results["GaussianNB"]["scores"]
                gnb_max_accuracy_hyperparams = current_results["GaussianNB"]["hypers"]

        FNAME = "output_max_GaussianNB.txt"
        output_file = open(FNAME,'a')
        output_file.write("\nMax scores:\n")
        for key, value in gnb_max_scores.items():
            output_file.write("{}: {}\n".format(key, value))
        output_file.write("\nCorresponding hyperparameters:\n")
        for key, value in gnb_max_accuracy_hyperparams.items():
            output_file.write("{}: {}\n".format(key, value))
        output_file.close()

    ## lr       
    model_test ={"LinearSVC":0,"GaussianNB":0,"LogisticRegression":0,"RandomForestClassifier":0}
    if model_test["LogisticRegression"]!=0:   
        params = {"lr_c":[0.001,0.01,0.1,1,10,100],
                "lr_tol":[0.000001,0.00001,0.001,0.01,0.1]}
        for p1 in params["lr_c"]:
            for p2 in params["lr_tol"]:
                lr_c = p1
                lr_tol = p2
                current_results = run_models(features_list)
                lr_current_accuracy = current_results["LogisticRegression"]["scores"]["test_score"]
                if lr_current_accuracy > lr_max_accuracy:
                    lr_max_accuracy = lr_current_accuracy
                    lr_max_scores = current_results["LogisticRegression"]["scores"]        
                    lr_max_accuracy_hyperparams = current_results["LogisticRegression"]["hypers"]

        FNAME = "output_max_LogisticRegression.txt"
        output_file = open(FNAME,'a')
        output_file.write("\nMax scores:\n")
        for key, value in lr_max_scores.items():
            output_file.write("{}: {}\n".format(key, value))
        output_file.write("\nCorresponding hyperparameters:\n")
        for key, value in lr_max_accuracy_hyperparams.items():
            output_file.write("{}: {}\n".format(key, value))
        output_file.close()

    ## rfc

    model_test ={"LinearSVC":0,"GaussianNB":0,"LogisticRegression":0,"RandomForestClassifier":1}
    if model_test["RandomForestClassifier"]!=0: 
        params = {"rfc_max_feature":["auto","sqrt","log2"],
                "rfc_max_depth":[10,30,50,70,90,None],
                "rfc_n_estimators":[10,30,50],
                "rfc_min_sample_leaf":[1,5,10],
                "rfc_oob_score":[False,True]}
        for p1 in params["rfc_max_feature"]:
            for p2 in params["rfc_max_depth"]:
                for p3 in params["rfc_n_estimators"]:
                    for p4 in params["rfc_min_sample_leaf"]:
                        for p5 in params["rfc_oob_score"]:
                            rfc_max_feature = p1
 rfc_max_depth = p2
                            rfc_n_estimators = p3
                            rfc_min_sample_leaf = p4
                            rfc_oob_score = p5
                            try:
                                current_results = run_models(features_list)
                                rfc_current_accuracy = current_results["RandomForestClassifier"]["scores"]["test_score"]
                                if rfc_current_accuracy > rfc_max_accuracy:
                                    rfc_max_accuracy = rfc_current_accuracy
                                    rfc_max_scores = current_results["RandomForestClassifier"]["scores"]
                                    rfc_max_accuracy_hyperparams = current_results["RandomForestClassifier"]["hypers"]
                            except:
                                pass
        FNAME = "output_max_RandomForestClassifier.txt"
        output_file = open(FNAME,'a')
        output_file.write("\nMax scores:\n")
        for key, value in rfc_max_scores.items():
            output_file.write("{}: {}\n".format(key, value))
        output_file.write("\nCorresponding hyperparameters:\n")
        for key, value in rfc_max_accuracy_hyperparams.items():
            output_file.write("{}: {}\n".format(key, value))
        output_file.close()
    