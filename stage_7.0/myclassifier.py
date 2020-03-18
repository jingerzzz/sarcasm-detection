import numpy as np
import pandas as pd
import string
import re
import os
import json

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt



def tokenize(input_text):
    output_list = input_text.split()
    return output_list

def words_replace(input_text,words_pair_list):
    for words_pair in words_pair_list:
        input_text.replace(words_pair[0],words_pair[1])
    return input_text

def better_tokenize(input_text):
    ## convert words into lower case
    input_text = input_text.lower()
    ## replace abbreviation
    words_pair_list = [["wanna","want to"],
                        ["gonna","going to"]
                        ]
    input_text = words_replace(input_text,words_pair_list)
    # separate words by space
    output_list = input_text.split()
    ## remove emtpy string
    result = [token for token in output_list if token!=""]
    ## remove stop words
    stop_words_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    for word in output_list:
        if word in stop_words_list:
            output_list.remove(word)

    ## remove punctuations
    table = str.maketrans('', '', string.punctuation)
    output_list = [word.translate(table) for word in output_list]
    ## bigram
    bigram_output_list = []
    for i in range(len(output_list)-1):
        two_gram = output_list[i]+" "+output_list[i+1]
        bigram_output_list.append(two_gram)

    return output_list

def train(X_train,y_train,smoothing_alpha=0):
    class_vocab_num_dict = {"pos":{},"neg":{}}
    vocab_set = set([])
    ## calculate the counts of each word under different labels, and create the whole vocabulary
    for i in range(len(X_train)):
        piece = X_train[i]
        word_list = tokenize(piece)
        label = y_train[i]
        for token in word_list:
            vocab_set.add(token)
            if label==1:
                if token in class_vocab_num_dict["pos"].keys():
                    class_vocab_num_dict["pos"][token]+=1
                else:
                    class_vocab_num_dict["pos"][token]=1
            elif label==0:
                if token in class_vocab_num_dict["neg"].keys():
                    class_vocab_num_dict["neg"][token]+=1
                else:
                    class_vocab_num_dict["neg"][token]=1
    ## calculate the frequencies of different labels, which is P(y)
    count_pos=0
    count_neg=0
    for i in range(len(y_train)):
        if y_train[i]==1:
            count_pos+=1
        else:
            count_neg+=1
    freq_pos=count_pos/(count_pos+count_neg)
    freq_neg=count_neg/(count_pos+count_neg)    
    ## count the total words under different labels
    pos_word_count = 0
    neg_word_count = 0
    for word, count in class_vocab_num_dict["pos"].items():
        pos_word_count += count
    for word, count in class_vocab_num_dict["neg"].items():
        neg_word_count += count
    # print(pos_word_count)
    # print(neg_word_count)
    class_vocab_freq_dict = class_vocab_num_dict
    ## calculate the frequencies of each word under different laels, which is P(x|y).
    size_vocab = len(vocab_set)
    for word, count in class_vocab_freq_dict["pos"].items():
        class_vocab_freq_dict["pos"][word]= (smoothing_alpha+count)/(pos_word_count + smoothing_alpha*size_vocab)
    for word, count in class_vocab_freq_dict["neg"].items():
        class_vocab_freq_dict["neg"][word]= (smoothing_alpha+count)/(neg_word_count + smoothing_alpha*size_vocab)

    ## calcualte the frequencies of new words that have been never observed in training data
    new_word_freq_dict = {"pos":0,"neg":0}
    new_word_freq_dict["pos"] = (smoothing_alpha)/(pos_word_count + smoothing_alpha*size_vocab)
    new_word_freq_dict["neg"] = (smoothing_alpha)/(neg_word_count + smoothing_alpha*size_vocab)
    

    train_NBC = [freq_pos, freq_neg, class_vocab_freq_dict,new_word_freq_dict]## NBC stands for Naive Bayes Classifier


    # with open('freq.txt', 'w') as json_file:
    #     json.dump(class_vocab_freq_dict, json_file)
    # print(len(vocab_set))
    # print(size_vocab)
    return train_NBC

def classify(tokenized_document,trained_NBC):## NBC stands for Naive Bayes Classifier
    posterior_pos_prob = 1
    posterior_neg_prob = 1
    freq_pos = trained_NBC[0]
    freq_neg = trained_NBC[1]
    class_vocab_freq_dict = trained_NBC[2]
    new_word_freq_dict = trained_NBC[3]
    for word in tokenized_document:
        ## calculate the posterior probability of being labeled as pos
        if word in class_vocab_freq_dict["pos"].keys():
            posterior_pos_prob = posterior_pos_prob*class_vocab_freq_dict["pos"][word]
        else:
            posterior_pos_prob = posterior_pos_prob*new_word_freq_dict["pos"]
        ## calculate the posterior probability of being labeled as pos
        if word in class_vocab_freq_dict["neg"].keys():
            posterior_neg_prob = posterior_neg_prob*class_vocab_freq_dict["neg"][word]
        else:
            posterior_neg_prob = posterior_neg_prob*new_word_freq_dict["neg"]
    posterior_pos_prob = posterior_pos_prob*freq_pos
    posterior_neg_prob = posterior_neg_prob*freq_neg

    classify_result = 0
    if posterior_pos_prob>=posterior_neg_prob:
        classify_result=1
    else:
        classify_result=0
    return classify_result

def myplot(X_train,y_train,X_dev,y_dev):
    alpha_list = np.arange(0,1.02,0.02)
    f1_list = []
    accuracy_list = []

    for alpha in alpha_list:
        trained_NBC = train(X_train, y_train,smoothing_alpha=alpha)
        y_pred = []
        for piece in X_dev:
            tokenized_piece = tokenize(piece)
            piece_pred = classify(tokenized_piece,trained_NBC)
            y_pred.append(piece_pred)
        tn, fp, fn, tp = confusion_matrix(y_dev,y_pred).ravel()
        accuracy_score = (tp+tn)/(tp+tn+fp+fn)
        precision_score = tp / (tp + fp)
        recall_score = tp / (tp + fn)
        f1_score = 2*precision_score*recall_score/(precision_score+recall_score) 
        f1_list.append(f1_score)
        accuracy_list.append(accuracy_score)
    plt.plot(alpha_list,accuracy_list)
    print(accuracy_list)
    plt.xlabel("smoothing_alpha")
    plt.ylabel("accuracy")
    plt.show()

def myvalid(X_train,y_train,X_dev,y_dev):
    trained_NBC = train(X_train, y_train,smoothing_alpha=20)
    
    y_pred = []
    index = 0
    for piece in X_dev:
        index+=1
        tokenized_piece = tokenize(piece)
        # if index<50:
            # print(tokenized_piece)
        piece_pred = classify(tokenized_piece,trained_NBC)
        y_pred.append(piece_pred)
    tn, fp, fn, tp = confusion_matrix(y_dev,y_pred).ravel()
    accuracy_score = (tp+tn)/(tp+tn+fp+fn)
    precision_score = tp / (tp + fp)
    recall_score = tp / (tp + fn)
    f1_score = 2*precision_score*recall_score/(precision_score+recall_score) 
    print("f1_score:{}\naccuracy:{}".format(f1_score,accuracy_score))
    

def mypred(X_train,y_train,X_test):
    trained_NBC = train(X_train, y_train,smoothing_alpha=0.9)
    y_pred = []
    for piece in X_test:
        tokenized_piece = tokenize(piece)
        piece_pred = classify(tokenized_piece,trained_NBC)
        y_pred.append(piece_pred)
    output_df = pd.DataFrame(data = y_pred, columns = ['Category'])
    output_df.index.name="Id"
    output_df.to_csv("y_pred_NB.csv")
    
    
#### main 

## load data
train_df = pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
train_df = train_df.rename(columns={train_df.columns[0]:"is_sarcastic", train_df.columns[1]:"comment", train_df.columns[2]:"parent_comment"})
test_df = pd.read_csv('labeled_test.csv',delimiter=',',encoding='utf-8')
test_df = test_df.rename(columns={test_df.columns[0]:"id", test_df.columns[1]:"comment", test_df.columns[2]:"parent_comment",test_df.columns[3]:"is_sarcastic"})
train_df = train_df.dropna()
test_df = test_df.dropna()




# Relacing special symbols and digits in comment column
# re stands for Regular Expression

train_df['comment'] = train_df['comment']

features_train = train_df['comment']
features_test = test_df['comment']
labels_train = train_df['is_sarcastic']
labels_test = test_df['is_sarcastic']

X_train = features_train.to_list()
X_test = features_test.to_list()
y_train = labels_train.tolist()
y_test = labels_test.tolist()

# X_train=X_train+X_dev
# y_train=y_train+y_dev
# myplot(X_train,y_train,X_test,y_test)
myvalid(X_train,y_train,X_test,y_test)
# mypred(X_test)