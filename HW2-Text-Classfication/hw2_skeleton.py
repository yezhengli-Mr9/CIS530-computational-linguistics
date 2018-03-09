
# coding: utf-8

# In[1]:


#############################################################
## ASSIGNMENT 2 CODE SKELETON
## RELEASED: 1/17/2018
## DUE: 1/24/2018
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
import re
#-------
#yezheng:
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from string import ascii_lowercase
letter_sele = 'aeiou-'#'aeiouthn' #ascii_lowercase#
#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true, debug = False):
    # deal with npdarray
    y_pred = list(y_pred)
    y_true = list(y_true)
    #---------
    n = len(y_pred);
    y_pred = [0 if v is None else v for v in y_pred]# deal with None type
    y_true = [0 if v is None else v for v in y_true]# deal with None type
    true_positive = sum(y_pred[i]* y_true[i] for i in range(n))
    if (0 == sum(y_pred)): return 0
    return true_positive*1.0/sum(y_pred)
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    # deal with npdarray
    y_pred = list(y_pred)
    y_true = list(y_true)
    #---------
    n = len(y_pred);
    y_pred = list(map(int,[1 == l for l in y_pred]))# deal with None type
    y_true = list(map(int,[1 == l for l in y_true]))# deal with None type
    true_positive = sum(y_pred[i]*y_true[i] for i in range(n))
    if 0 == sum(y_true): return 0
    return true_positive*1.0/sum(y_true)
    

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    precision = get_precision(y_pred, y_true);
    if (0 == precision): return 0
    recall= get_recall(y_pred, y_true);
    if (0 == recall): return 0
    beta = 1.0;
    # print("get_fscore:",(beta**2*precision+recall))
    fscore = (beta**2+1)*precision*recall/(beta**2*precision+recall);
    return fscore

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file, debug = False):
    labels = []   
    words = []
    Lst_data = data_file
    if isinstance(data_file, str): Lst_data = [data_file] # isinstance(data_file,basestring) for python2
    for fname in Lst_data:
        with open(fname, 'rt', encoding="utf8") as f:
            Lines = f.readlines()
            Lines = Lines[1:]
            num_data = len(Lines)# remove first ele in Lines, remove last one 
            Lst_pos1 = [line[:-1].find('\t') for line in Lines]
            words += [Lines[i][:Lst_pos1[i]] for i in range(num_data)]
            Lst_pos2 = [Lines[i][(Lst_pos1[i]+1):-1].find('\t')+Lst_pos1[i]+1 for i in range(num_data )]
            for i in range(num_data ):
                # if debug:
                #     print(i,"----",Lines[i])
                #     print("#########",Lines[i][(Lst_pos1[i]+1):Lst_pos2[i]],"#######")
                if (re.match("^\d+?(\.\d+)?$",Lines[i][(Lst_pos1[i]+1):Lst_pos2[i]])): labels.append(int(Lines[i][(Lst_pos1[i]+1):Lst_pos2[i]]))
                else: labels.append(0) #None
    # if debug: print("load file DEBUG:",len(words),len(labels),labels)
        # labels = [int(Lines[i+1][(Lst_pos1[i]+1):Lst_pos2[i]]) for i in range(num_data)] 
    return words, labels

#-------------
# yezheng: default one
# ## Loads in the words and labels of one of the datasets
# def load_file(data_file):
#     words = []
#     labels = []   
#     with open(data_file, 'rt', encoding="utf8") as f:
#         i = 0
#         for line in f:
#             if i > 0:
#                 line_split = line[:-1].split("\t")
#                 words.append(line_split[0])
#                 labels.append(int(line_split[1]))
#             i += 1
#     return words, labels
#-------------
### 2.1: A very simple baseline

## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    words,labels = load_file(data_file)
    y_pred = [1] * len(words)
    precision = get_precision(y_pred, labels)
    recall = get_recall(y_pred, labels)
    fscore = get_fscore(y_pred, labels)
    return precision, recall, fscore


### 2.2: Word length thresholding

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file, plot_flag = False):
    ## YOUR CODE HERE
    # print("load training file")
    words,labels = load_file(training_file, True)
    Dict_len = defaultdict(set)
    for w in words: Dict_len[len(w)].add(w)
    # Evaluation depending on fscore
    Min = min(Dict_len)
    Max = max(Dict_len)
    # for Thres in range(Min,Max): 
    #     # print(list(map(int,[len(w) < Thres for w in words])))
    #     print(get_precision(list(map(int,[len(w) < Thres for w in words])), labels ) ) 
    #----------
    #yezheng: binary search is possible to improve the serach
    #plotting precision vs. recall 
    ThreRange =  range(Min,Max+1)
    FscoreL = dict([(get_fscore(list(map(int,[len(w) > Thres for w in words])), labels),Thres) for Thres in ThreRange])
    Thres_opt = FscoreL[max(FscoreL)]
    if plot_flag:
        print("Range of thresholds:",Min,"to",Max, " with optimal threshold:", Thres_opt+1)
        precisionL = [get_precision(list(map(int,[len(w) > Thres for w in words])), labels)for Thres in ThreRange]
        recallL = [get_recall(list(map(int,[len(w) > Thres for w in words])), labels) for Thres in ThreRange]
        Ret_plot = [precisionL,recallL]
        plt.plot(recallL, precisionL,'^r', label = 'Train')
        plt.title('word_length_threshold: precision-recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')

    Pred_opt = list(map(int,[len(w)> Thres_opt for w in words]))
    training_performance = [get_precision(Pred_opt, labels), get_recall(Pred_opt, labels), get_fscore(Pred_opt, labels)]
    del words
    del labels
    # print("load development file")
    words,labels = load_file(development_file)
    Pred_opt = list(map(int,[len(w)> Thres_opt for w in words]))
    if plot_flag: 
        precisionL = [get_precision(list(map(int,[len(w) > Thres for w in words])), labels)for Thres in ThreRange]
        recallL = [get_recall(list(map(int,[len(w) > Thres for w in words])), labels) for Thres in ThreRange]
        plt.plot(recallL, precisionL,'^g', label = 'Dev')
        plt.legend(loc='upper right')
        plt.show()
        Ret_plot += [precisionL,recallL]
    development_performance = [get_precision(Pred_opt, labels), get_recall(Pred_opt, labels), get_fscore(Pred_opt, labels)]
    if plot_flag: return training_performance, development_performance,Ret_plot
    return training_performance, development_performance


# In[2]:


### 2.3: Word frequency thresholding
## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   # counts = defaultdict(int) 
   counts = {}
   with gzip.open(ngram_counts_file, 'rt') as f: 
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set
def word_frequency_threshold(training_file, development_file, counts, plot_flag = False):
    words,labels = load_file(training_file)
#     Min = min(counts[w] for w in words if w in counts)
#     Max_ = max(counts[w] for w in words if w in counts)
#     Max = Max_#int((Max_ - Min) /(10**7)) + Min # manually do the test
#     ThreRange = range(Min,Max, int((Max-Min)/10))
    Min =int(19903896) #19903996#<- 19903896#<-19903906# <-19902396 #<- 19881406 #<- 19802396
    Max = 2* 19903996  - 19903896 
#     print("Max - Min",Max - Min) #158020
    ThreRange = range(Min,Max) # int((Max-Min)/3000)
#     ThreRange = range(Min,Max)
    FscoreL = dict([(get_fscore(list(map(int,[counts[w]< Thres if w in counts else 0 for w in words])), labels) ,Thres) for Thres in ThreRange])
    Thres_opt = FscoreL[max(FscoreL)]
    if plot_flag:
#         print("Range of thresholds:",Min,"to",Max, " with optimal threshold:", Thres_opt) # Dev has larger range
        Min = 137
        Max = 1120679362
        ThreRange = range(Min,Max,int((Max-Min)/1000))
        precisionL = [get_precision(list(map(int,[counts[w]< Thres if w in counts else 0 for w in words])), labels)  for Thres in ThreRange]
        recallL = [get_recall(list(map(int,[counts[w]< Thres if w in counts else 0 for w in words])), labels)  for Thres in ThreRange]
#         print("precisionL",precisionL)
#         print("recallL", recallL)
        plt.plot(recallL, precisionL,'.b',label = "Train")
        plt.title('word_frequency_threshold: precision-recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        Ret_plot = [precisionL,recallL]
    labels_pred = list(map(int,[counts[w]< Thres_opt if w in counts else 0 for w in words]))
    training_performance = [get_precision(labels_pred,labels), get_recall(labels_pred,labels), get_fscore(labels_pred,labels)]
    words,labels = load_file(development_file, True)
#     Min = min(counts[w] for w in words if w in counts)
#     Max_ = max(counts[w] for w in words if w in counts)
#     Max = Max_#int((Max_ - Min) /(10**7)) + Min # manually do the test
#     ThreRange = range(Min,Max, int((Max-Min)/10))
#     ThreRange = range(Min,Max, int((Max-Min)/1000))
    labels_pred = list(map(int,[counts[w]< Thres_opt if w in counts else 0 for w in words]))
    if plot_flag:
        Min = 137
        Max = 1120679362
        ThreRange = range(Min,Max,int((Max-Min)/1000))
        print("Range of thresholds:",Min,"to",Max, " with optimal threshold:", Thres_opt)
        precisionL = [get_precision(list(map(int,[counts[w]< Thres if w in counts else 0 for w in words])), labels)  for Thres in ThreRange]
        recallL = [get_recall(list(map(int,[counts[w]< Thres if w in counts else 0 for w in words])), labels)  for Thres in ThreRange]
#         print("precisionL",precisionL)
#         print("recallL", recallL)
        plt.plot(recallL, precisionL,'.y',label = "Dev")
        plt.legend(loc='upper right')
        plt.show()
        Ret_plot += [precisionL,recallL]
    development_performance = [get_precision(labels_pred,labels), get_recall(labels_pred,labels), get_fscore(labels_pred,labels)]
    if plot_flag: return training_performance, development_performance,Ret_plot
    return training_performance, development_performance


# In[3]:


### 2.4: Naive Bayes
from sklearn.naive_bayes import GaussianNB
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):    
    words,labels = load_file(training_file)
    labels_np = np.array(labels)
    X_features = np.array([[1.0*len(w), counts[w]] if w in counts else [1.0*len(w),0] for w in words])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    clf = GaussianNB(); clf.fit(X_features, labels_np)
    Y_pred_np = clf.predict(X_features)

    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    labels_np = np.array(labels)
    X_features = np.array([[1.0*len(w), counts[w]] if w in counts else [1.0*len(w),0] for w in words])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    Y_pred_np = clf.predict(X_features)
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    return training_performance, development_performance

### 2.5: Logistic Regression
from sklearn.linear_model import LogisticRegression, LinearRegression
## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    words,labels = load_file(training_file)
    labels_np = np.array(labels)
    X_features = np.array([[1.0*len(w), counts[w]] if w in counts else [1.0*len(w),0] for w in words])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    clf = LogisticRegression(); #penalty = 'l1'
    clf.fit(X_features, labels_np)
    Y_pred_np = clf.predict(X_features)

    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    labels_np = np.array(labels)
    X_features = np.array([[1.0*len(w), counts[w]] if w in counts else [1.0*len(w),0] for w in words])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    Y_pred_np = clf.predict(X_features)
    # print("Y_pred_np",Y_pred_np)
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    return training_performance, development_performance


# In[4]:


### 2.7: Build your own classifier
## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE
from syllables import count_syllables
# from nltk.corpus import wordnet as wn #wn.lemma(w).count(),
def preprocess_yezheng(words,labels, counts):
    Thres_opt_len = 6
    Thres_opt_freq = 19904037#<-19903996#<- 19903896#<-19903906# <-19902396 #<- 19881406 #<- 19802396
    # 1.0*len(w),
#     X_features = [[1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len], int(counts[w] < Thres_opt_freq), counts[w] ]+[w.count(alp) for alp in letter_sele] if w in counts else [1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len],1,1120679362]+[w.count(alp) for alp in letter_sele] for w in words]
    # best
    X_features = np.array([[1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len], int(counts[w] < Thres_opt_freq), counts[w] ]+[w.count(alp) for alp in letter_sele] if w in counts else [1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len],1,1120679362]+[w.count(alp) for alp in letter_sele] for w in words]) 
#     X_features = np.array([[1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len], int(counts[w] < Thres_opt_freq) ]+[w.count(alp) for alp in letter_sele] if w in counts else [1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len],1]+[w.count(alp) for alp in letter_sele] for w in words])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
#     X_features = np.array([np.concatenate((row,np.convolve(row,row))) for row in X_features])
#     scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    return X_features, np.array(labels)


# In[5]:


from sklearn.ensemble import RandomForestClassifier

def improved_naive_bayes(training_file, development_file, counts,show_err_words_flag = False):    
    words,labels = load_file(training_file)
    labels_np = np.array(labels)
    X_features, labels_np = preprocess_yezheng(words, labels,counts)
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    clf = GaussianNB(); clf.fit(X_features, labels_np)
    Y_pred_np = clf.predict(X_features)
    if show_err_words_flag: L_train = [X_features,words,list(Y_pred_np),labels]
#         print("Naive Bayes (improved)")
#         Y_lst = list(Y_pred_np)
#         print("Train:",[words[i] for i in range(len(words)) if not Ylst[i] == labels[i]])
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    labels_np = np.array(labels)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    Y_pred_np = clf.predict(X_features)
    if show_err_words_flag: L_dev = [X_features,words,list(Y_pred_np),labels]
#     if show_err_words_flag: 
#         print("Dev:",[words[i] for i in range(len(words)) if not Ylst[i] == labels[i]])
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    if not show_err_words_flag: return training_performance, development_performance  
    else: return clf,L_train, L_dev

def improved_log_regression(training_file, development_file, counts):
    words,labels = load_file(training_file)
    labels_np = np.array(labels)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)    
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    clf = LogisticRegression(); #penalty = 'l1'
    clf.fit(X_features, labels_np)
    Y_pred_np = clf.predict(X_features)
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    labels_np = np.array(labels)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    Y_pred_np = clf.predict(X_features)
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    return training_performance, development_performance

def random_forest(training_file, development_file, counts):
    words,labels = load_file(training_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    clf = RandomForestClassifier(max_depth=2, random_state=0); clf.fit(X_features, labels_np) # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    Y_pred_np = clf.predict(X_features)
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    Y_pred_np = clf.predict(X_features)
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    return training_performance, development_performance

from sklearn.tree import DecisionTreeClassifier 
def decision_tree(training_file, development_file, counts):
    words,labels = load_file(training_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0); clf.fit(X_features, labels_np) # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    Y_pred_np = clf.predict(X_features)

    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    Y_pred_np = clf.predict(X_features)
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    return training_performance, development_performance

from sklearn.svm import SVC, LinearSVC
def SVM_SVC(training_file, development_file, counts,show_err_words_flag = False):
    words,labels = load_file(training_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    clf = SVC(); clf.fit(X_features, labels_np) 
    Y_pred_np = clf.predict(X_features)
    if show_err_words_flag: L_train = [X_features,words,list(Y_pred_np),labels]
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    Y_pred_np = clf.predict(X_features)
    if show_err_words_flag: L_dev = [X_features,words,list(Y_pred_np),labels]
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    if not show_err_words_flag: return training_performance, development_performance  
    else: return clf,L_train, L_dev

def SVM_LinearSVC(training_file, development_file, counts):
    words,labels = load_file(training_file)  
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    clf = LinearSVC(); clf.fit(X_features, labels_np) # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    Y_pred_np = clf.predict(X_features)
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    Y_pred_np = clf.predict(X_features)
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    return training_performance, development_performance

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
def QDA(training_file, development_file, counts):
    words,labels = load_file(training_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    clf = QuadraticDiscriminantAnalysis(); clf.fit(X_features, labels_np) # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    Y_pred_np = clf.predict(X_features)

    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    Y_pred_np = clf.predict(X_features)
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    return training_performance, development_performance

def LDA(training_file, development_file, counts):
    words,labels = load_file(training_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    clf = LinearDiscriminantAnalysis(); clf.fit(X_features, labels_np) # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    Y_pred_np = clf.predict(X_features)
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    labels_np = np.array(labels)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    Y_pred_np = clf.predict(X_features)
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    return training_performance, development_performance
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
def AdaBoost(training_file, development_file, counts):
    words,labels = load_file(training_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    clf = AdaBoostClassifier(); clf.fit(X_features, labels_np) # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    Y_pred_np = clf.predict(X_features)
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    labels_np = np.array(labels)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    Y_pred_np = clf.predict(X_features)
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    return training_performance, development_performance
def GradBoost(training_file, development_file, counts, show_err_words_flag = False):
    words,labels = load_file(training_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    clf = GradientBoostingClassifier(); clf.fit(X_features, labels_np) # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    Y_pred_np = clf.predict(X_features)
    if show_err_words_flag: L_train = [X_features,words,list(Y_pred_np),labels]
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    Y_pred_np = clf.predict(X_features)
    if show_err_words_flag: L_dev = [X_features,words,list(Y_pred_np),labels]
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    if not show_err_words_flag: return training_performance, development_performance  
    else: return clf,L_train, L_dev


# In[6]:


# training_file = "data/complex_words_training.txt"
# development_file = "data/complex_words_development.txt"
# test_file = "data/complex_words_test_unlabeled.txt"
# ngram_counts_file = "ngram_counts.txt.gz"
# counts = load_ngram_counts(ngram_counts_file)


# In[7]:


if __name__ == "__main__":
    import time
    T0 = time.time()
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
    all_complex(training_file)
    import datetime
    print("HW2-Writeup", datetime.datetime.now())
    print("Yezheng Li, Daizhen Li")
    print("-------------------------")
    print("\033[1;32;10mBaselines\033[0m")
    print("\033[1;32;10mAll-complex Baseline:\033[0m")
    p1,r1,f1 = all_complex(training_file); print("Train: precision",p1,"recall",r1,"F-score",f1)
    p2,r2,f2 = all_complex(development_file); print("Dev: precision",p2,"recall",r2,"F-score",f2)
    print("\033[1;32;10mWord-length Baseline:\033[0m")
    result1, result2,Ret_plot_len = word_length_threshold(training_file,development_file ,True)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("Train: precision",p1,"recall",r1,"F-score",f1)
    print("Dev: precision",p2,"recall",r2,"F-score",f2)
    print("\033[1;32;10mWord-frequency Baseline:\033[0m")
    result1, result2, Ret_plot_freq= word_frequency_threshold(training_file,development_file,counts, True)
    p1_wf,r1_wf,f1_wf = result1; p2_wf,r2_wf,f2_wf= result2
    print("Train: precision",p1_wf,"recall",r1_wf,"F-score",f1_wf)
    print("Dev: precision",p2_wf,"recall",r2_wf,"F-score",f2_wf)
    print("\033[1;32;10mPlot Precison-Recall curve for various thresholds for both baselines together:\033[0m")
    [pt_len, rt_len, pd_len, rd_len]= Ret_plot_len
    [pt_fq, rt_fq, pd_fq, rd_fq]= Ret_plot_freq
    plt.plot(rt_len,pt_len,'^g',label = "Train (length)")
    plt.plot(rd_len,pd_len, '^r',label = "Dev (length)")
    plt.plot( rt_fq,pt_fq,'.b',label = "Train (frequency)")
    plt.plot(rd_fq,pd_fq, '.y',label = "Dev (frequency)")
    plt.legend(loc='upper right')
    plt.title('word_length_threshold\& word_frequency_threshold: precision-recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    print("\033[1;32;10mWhich classifier looks better on average?\033[0m I think word_length_threshold is better.")
    print("-------------------------------------------------------------------------------------------------------")
    print("\033[1;32;10mNaive Bayes:\033[0m")
    result1, result2= naive_bayes(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("Train: precision",p1,"recall",r1,"F-score",f1)
    print("Dev: precision",p2,"recall",r2,"F-score",f2)
    print("\033[1;32;10mLogistic regression:\033[0m")
    result1, result2= logistic_regression(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("Train: precision",p1,"recall",r1,"F-score",f1)
    print("Dev: precision",p2,"recall",r2,"F-score",f2)
#     print("-------------------------")
    print("\033[1;32;10mAdd a paragraph to your write up that discusses which model performed better on this task.\033[0m")
    print("Although naive Bayes performslightly better in term of F-score, I think logistic regression have more balanced performance between precision and recall.")
    print("-------------------------------------------------------------------------------------------------------")

    print("\033[1;32;10mBuild your own model\033[0m")
    print("\033[1;32;10mPlease include a description of all features that you tried (not including length and frequency).\033[0m")
    print("Besides ength and frequency (with and without thresholds -- 4 features), I tried: [see preprocess_yezheng(...)]")
    print("--- count of character 'aeiou-' (I also tried string.ascii_lowercase,etc. but the latter has worse performance) -- 6 features; ")
    print("--- count_syllables(...) -- 1 features")
    print("--- convolution of feature with itself: np.convolve(X,X) -- does not work")
    print("In all, there are 11 features")
    print("\033[1;32;10mPlease include a description of all models that you tried.\033[0m")
    print("Besides improved Naive Bayes and improved logistic regression (with penality l1 or l2), I tried:") 
    print("random forest, decision tree, svm.SVC, svm.LinearSVC, LDA (linear discriminant analysis), QDA (Quadratic Discriminant Analysis). ")
    print("\033[1;32;10mPerform a detailed error analysis of your models.\033[0m")  
    print("See table below for a summary. \033[1;31;10mRed\033[0m highlights are best performances (it varies depending on different experiments, I just highlight best performances in common (for various experiments.)).")
    print("Train:\t\t\t\t\tDev")
    print("precision\trecall\tF-score\t\tprecision\trecall\tF-score\tclassifier")
    print("Baselines--------------------------------------------------------------------------------------------------------")
    p1,r1,f1 = all_complex(training_file);p2,r2,f2 = all_complex(development_file);
    print("%.9f %.9f %.9f|%.9f %.9f %.9f All-complex" % (p1,r1,f1,p2,r2,f2))
    result1, result2= word_length_threshold(training_file,development_file)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f Word-length" % (p1,r1,f1,p2,r2,f2))
    #     result1, result2= word_frequency_threshold(training_file,development_file, counts)
    #     p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f Word-frequency" % (p1_wf,r1_wf,f1_wf,p2_wf,r2_wf,f2_wf))
    print("--------------------------------------------------------------------------------------------------------")
    result1, result2= naive_bayes(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f Naive Bayes" % (p1,r1,f1,p2,r2,f2))
    result1, result2= logistic_regression(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f Logistic regression" % (p1,r1,f1,p2,r2,f2))
    print("--------------------------------------------------------------------------------------------------------")
    result1, result2= improved_naive_bayes(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f Naive Bayes (improved)" % (p1,r1,f1,p2,r2,f2))
    result1, result2= improved_log_regression(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f Logistic regression (improved)" % (p1,r1,f1,p2,r2,f2))
    result1, result2= random_forest(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f Random forest" % (p1,r1,f1,p2,r2,f2))
    result1, result2= decision_tree(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f Decision tree" % (p1,r1,f1,p2,r2,f2))
    result1, result2= SVM_SVC(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f \033[1;31;10m%.9f\033[0m|%.9f %.9f \033[1;31;10m%.9f\033[0m SVM SVC" % (p1,r1,f1,p2,r2,f2))
    result1, result2= SVM_LinearSVC(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f SVM linear SVC" % (p1,r1,f1,p2,r2,f2))
    result1, result2= QDA(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f QDA" % (p1,r1,f1,p2,r2,f2))
    result1, result2= LDA(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f LDA" % (p1,r1,f1,p2,r2,f2))
    result1, result2= AdaBoost(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f %.9f|%.9f %.9f %.9f AdaBoost" % (p1,r1,f1,p2,r2,f2))
    result1, result2= GradBoost(training_file,development_file,counts)
    p1,r1,f1 = result1; p2,r2,f2= result2
    print("%.9f %.9f \033[1;31;10m%.9f\033[0m|%.9f %.9f \033[1;31;10m%.9f\033[0m Gradient Boost" % (p1,r1,f1,p2,r2,f2))
    print("--------------------------------------------------------------------------------------------------------")
    print("\033[1;32;10mAnalyze your model\033[0m")
    print("\033[1;32;10mAn important part of text classification tasks is to determine what your model is getting correct, and what your model is getting wrong. For this problem, you must train your best model on the training data, and report the precision, recall, and f-score on the development data.\033[0m")
    print("As a result, I think our best model is \033[1;31;10mSVM SVC\033[0m as well as \033[1;31;10mGradient Boost\033[0m (Although GradBoost generally outperforms SVM SVC in both Train and Dev with respect to F score, it is surprising that when it comes to Leaderboard, SVM SVC results in better ranking (typically with 2\% better Fscore).)")
    print("\033[1;32;10mGive several examples of words on which your best model performs well. Also give examples of words which your best model performs poorly on, and identify at least TWO categories of words on which your model is making errors.\033[0m")
    clf,L_train,L_dev = GradBoost(training_file,development_file,counts, show_err_words_flag = True)
    X, w,l_pred, l_true = L_train
    print("-------------------------------------")
    print("\033[1;31;10mGradient Boost: \033[1;32;10mcorrect prediction:\033[0m")
    print("Examples of true positive", [w[i] for i in range(len(l_true)) if 1 == l_pred[i] and 1 == l_true[i]][:10])
    print("Examples of false negative", [w[i] for i in range(len(l_true)) if 0 == l_pred[i] and 0 == l_true[i]][:10])
    print("\033[1;32;10mIncorrect prediction:\033[0m")
    print("Examples of false positive (i.e. not complex, but are predicted to be)", [w[i] for i in range(len(l_true)) if 1 == l_pred[i] and 0 ==  l_true[i]][:10])
    print("Examples of true negative (i.e. complex, but are predicted not to be)", [w[i] for i in range(len(l_true)) if 0 == l_pred[i] and 1 == l_true[i]][:10])
    #     print("clf.predict_log_proba(X)",clf.predict_log_proba(X))
    clf,L_train,L_dev = SVM_SVC(training_file,development_file,counts, show_err_words_flag = True)
    X,w, l_pred, l_true = L_train
    #     print("clf.decision_function(X)",clf.decision_function(X))
    print("\033[1;31;10mSVM SVC: \033[1;32;10mcorrect prediction:\033[0m")
    num_prt =8
    print("Examples of true positive", [w[i] for i in range(len(l_true)) if 1 == l_pred[i] and 1 == l_true[i]][:num_prt])
    print("Examples of false negative", [w[i] for i in range(len(l_true)) if 0 == l_pred[i] and 0 == l_true[i]][:num_prt])
    print("\033[1;32;10mIncorrect prediction:\033[0m")
    print("Examples of false positive (i.e. not complex, but are predicted to be)", [w[i] for i in range(len(l_true)) if 1 == l_pred[i] and 0 ==  l_true[i]][:num_prt])
    print("Examples of true negative (i.e. complex, but are predicted not to be)", [w[i] for i in range(len(l_true)) if 0 == l_pred[i] and 1 == l_true[i]][:num_prt])
    print("-------------------------------------")
    print("Time elapsed:", time.time() - T0) # time evaluation -- though improvement of precision(), recall(), fscore()


# In[15]:


# leaderboard output

unlabeled_file = "data/complex_words_test_unlabeled.txt"
def load_file2(data_file):
#     labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        Lines = f.readlines()
        Lines = Lines[1:]
        num_data = len(Lines)# remove first ele in Lines, remove last one 
        Lst_pos1 = [line[:-1].find('\t') for line in Lines]
        words = [Lines[i][:Lst_pos1[i]] for i in range(num_data)]
        Lst_pos2 = [Lines[i][(Lst_pos1[i]+1):-1].find('\t')+Lst_pos1[i]+1 for i in range(num_data )]
#         for i in range(num_data ):
#             # if debug:
#             #     print(i,"----",Lines[i])
#             #     print("#########",Lines[i][(Lst_pos1[i]+1):Lst_pos2[i]],"#######")
#             if (re.match("^\d+?(\.\d+)?$",Lines[i][(Lst_pos1[i]+1):Lst_pos2[i]])): labels.append(int(Lines[i][(Lst_pos1[i]+1):Lst_pos2[i]]))
#             else: labels.append(None)
    # if debug: print("load file DEBUG:",len(words),len(labels),labels)
        # labels = [int(Lines[i+1][(Lst_pos1[i]+1):Lst_pos2[i]]) for i in range(num_data)] 
        return words

def improved_naive_bayes_unlabeled_output(training_lst, unlabeled_file, counts):
    words,labels = load_file(training_lst)
    labels_np = np.array(labels)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
    clf = GaussianNB(); clf.fit(X_features, labels_np)
    Y_pred_np = clf.predict(X_features)
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words = load_file2(unlabeled_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts) #labels are not useful here
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    Y_pred_np = clf.predict(X_features)
    fname = "test_labels.txt"
    with open(fname, 'wt', encoding="utf8") as f: 
        for l in Y_pred_np: f.write(str(l)+'\n')
    return training_performance

print("-------------------------")
print("Naive Bayes (improved) unlabeled output:")
p,r,f= improved_naive_bayes_unlabeled_output([training_file,development_file],unlabeled_file,counts)
print("Train: precision",p,"recall",r,"F-score",f)


def SVM_SVC_unlabeled_output(training_lst, unlabeled_file, counts):
    words,labels = load_file(training_lst)
    labels_np = np.array(labels)
    X_features, labels_np = preprocess_yezheng(words, labels, counts)
#     print([len(fe) for fe in X_features])
#     print([row for row in X_features if None in row])
#     print([i for i in list(labels_np) if None == i])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    clf = SVC(); clf.fit(X_features, labels_np)
    Y_pred_np = clf.predict(X_features)
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words = load_file2(unlabeled_file)
    X_features, labels_np = preprocess_yezheng(words, labels, counts) #labels are not useful here
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    Y_pred_np = clf.predict(X_features)
    fname = "test_labels.txt"
    with open(fname, 'wt', encoding="utf8") as f: 
        for l in Y_pred_np: f.write(str(l)+'\n')
    return training_performance

print("-------------------------")
print("SVM SVC unlabeled output:")
p,r,f= SVM_SVC_unlabeled_output([training_file,development_file],unlabeled_file,counts)
print("Train: precision",p,"recall",r,"F-score",f)


# def AdaBoost_unlabeled_output(training_lst, unlabeled_file, counts):
#     words,labels = load_file(training_lst)
#     labels_np = np.array(labels)
#     X_features, labels_np = preprocess_yezheng(words, labels, counts)
#     clf = AdaBoostClassifier(); clf.fit(X_features, labels_np)
#     Y_pred_np = clf.predict(X_features)
#     training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
#     words = load_file2(unlabeled_file)
#     X_features, labels_np = preprocess_yezheng(words, labels, counts) #labels are not useful here
#     Y_pred_np = clf.predict(X_features)
#     fname = "test_labels.txt"
#     with open(fname, 'wt', encoding="utf8") as f: 
#         for l in Y_pred_np: f.write(str(l)+'\n')
#     return training_performance

# print("-------------------------")
# print("AdaBoost unlabeled output:")
# p,r,f= GradBoost_unlabeled_output([training_file,development_file],unlabeled_file,counts)
# print("Train: precision",p,"recall",r,"F-score",f)
# def GradBoost_unlabeled_output(training_lst, unlabeled_file, counts):
#     words,labels = load_file(training_lst)
#     labels_np = np.array(labels)
#     X_features, labels_np = preprocess_yezheng(words, labels, counts)
#     clf = GradientBoostingClassifier(); clf.fit(X_features, labels_np)
#     Y_pred_np = clf.predict(X_features)
#     training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
#     words = load_file2(unlabeled_file)
#     X_features, labels_np = preprocess_yezheng(words, labels, counts) #labels are not useful here
#     Y_pred_np = clf.predict(X_features)
#     fname = "test_labels.txt"
#     with open(fname, 'wt', encoding="utf8") as f: 
#         for l in Y_pred_np: f.write(str(l)+'\n')
#     return training_performance

# print("-------------------------")
# print("GradBoost unlabeled output:")
# p,r,f= GradBoost_unlabeled_output([training_file,development_file],unlabeled_file,counts)
# print("Train: precision",p,"recall",r,"F-score",f)
print("-------------------------------------")
print("Time elapsed:", time.time() - T0) # time evaluation -- though improvement of precision(), recall(), fscore()

