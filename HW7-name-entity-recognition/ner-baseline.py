
# coding: utf-8

# In[1]:


from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
# import sklearn_crfsuite
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import numpy as np
import scipy as sp
import sklearn
import time
from itertools import groupby


# In[2]:


# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, o, tag): 
    #if just word 61.08 in F1-score (window size = 2,3; optimal window size = 2)
    #if just word_prefix 36.54 in F1-score (window size = 2, 10,20; optimal window size = 20)
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    
    o = str(o)
    shape_feature = ['X' if ch.isupper() else ch for ch in word]
    shape_feature = "".join(['x' if ch.islower() else ch for ch in shape_feature]) #if just shape_feature 36.19 in F1-score (best amongst window size = 2,3,5,10; optimal window size = 10)
    short_shape_feature = "".join([x[0] for x in groupby(shape_feature)]) # if just short_shape_feature 29.43 in F1-score (window size = 2,5,10; optimal window size = 5)
    features = [
        (o + 'word', word)
#         , (o+"PoS", tag) #yezheng: I like tags
#         (o + 'word', shape_feature),
#         (o + 'word', short_shape_feature)
#         (o + 'word_prefix', word[0])
        
        # TODO: add more features here.
    ]
    return features

def getfeats_tag(word, o, tag): 
    #if just word 61.08 in F1-score (window size = 2,3; optimal window size = 2)
    #if just word_prefix 36.54 in F1-score (window size = 2, 10,20; optimal window size = 20)
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    shape_feature = ['X' if ch.isupper() else ch for ch in word]
    shape_feature = "".join(['x' if ch.islower() else ch for ch in shape_feature]) #if just shape_feature 36.19 in F1-score (best amongst window size = 2,3,5,10; optimal window size = 10)
    short_shape_feature = "".join([x[0] for x in groupby(shape_feature)]) # if just short_shape_feature 29.43 in F1-score (window size = 2,5,10; optimal window size = 5)
    features = [ (o+"PoS", tag) #yezheng: I like tags
#         (o + 'word_shape', shape_feature),
#         (o + 'word_short_shape', short_shape_feature)
#         (o + 'word_prefix', word[0])
        
        # TODO: add more features here.
    ]
    return features
    
    
def word2features(sent, i, DEBUG_FLAG = False):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    win_size = 4 # results2 results3
    win_size_tag = 10
    for o in range(-win_size,win_size+1):
#     for o in [-1,0,1]: 
        if i+o >= 0 and i+o < len(sent):
            featlist = getfeats(sent[i+o][0], o, sent[i+o][1])
            features.extend(featlist)
    for o in range(-win_size_tag,win_size_tag + 1):
        if i+o >= 0 and i+o < len(sent):
            featlist = getfeats_tag(sent[i+o][0], o, sent[i+o][1])
            features.extend(featlist)
    if DEBUG_FLAG: print("sent",sent,"i",i,"features",features)
    return dict(features)


# In[3]:


# if __name__ == "__main__":
# Load the training data
train_sents = list(conll2002.iob_sents('esp.train'))
#--------
# with open("content_train.txt",'w')  as f:
#     for sent in train_sents:
#         f.write(" ".join([ele[0] for ele in sent]) + '\n')
#--------
train_feats = []
train_labels = []

T0 = time.time()
for sent in train_sents:
    train_feats += [word2features(sent,i) for i in range(len(sent))]
    train_labels += [sent[i][-1] for i in range(len(sent))]
    
    
#--------    
# S = train_sents[0]
# for i in range(len(sent)): word2features(S,i, DEBUG_FLAG = True)
#--------
# sent [('Melbourne', 'NP', 'B-LOC'), ('(', 'Fpa', 'O'), ('Australia', 'NP', 'B-LOC'), (')', 'Fpt', 'O'), (',', 'Fc', 'O'), ('25', 'Z', 'O'), ('may', 'NC', 'O'), ('(', 'Fpa', 'O'), ('EFE', 'NC', 'B-ORG'), (')', 'Fpt', 'O'), ('.', 'Fp', 'O')] i 0 features [('0word', 'Melbourne'), ('1word', '('), ('2word', 'Australia')]
# sent [('Melbourne', 'NP', 'B-LOC'), ('(', 'Fpa', 'O'), ('Australia', 'NP', 'B-LOC'), (')', 'Fpt', 'O'), (',', 'Fc', 'O'), ('25', 'Z', 'O'), ('may', 'NC', 'O'), ('(', 'Fpa', 'O'), ('EFE', 'NC', 'B-ORG'), (')', 'Fpt', 'O'), ('.', 'Fp', 'O')] i 1 features [('-1word', 'Melbourne'), ('0word', '('), ('1word', 'Australia'), ('2word', ')')]
# sent [('Melbourne', 'NP', 'B-LOC'), ('(', 'Fpa', 'O'), ('Australia', 'NP', 'B-LOC'), (')', 'Fpt', 'O'), (',', 'Fc', 'O'), ('25', 'Z', 'O'), ('may', 'NC', 'O'), ('(', 'Fpa', 'O'), ('EFE', 'NC', 'B-ORG'), (')', 'Fpt', 'O'), ('.', 'Fp', 'O')] i 2 features [('-2word', 'Melbourne'), ('-1word', '('), ('0word', 'Australia'), ('1word', ')'), ('2word', ',')]


vectorizer = DictVectorizer()

X_train = vectorizer.fit_transform(train_feats) 
# print("X_train",X_train)
# print(X_train)
# X_train = X_train.toarray()
# scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_train); X_train = scaler.transform(X_train)
print("Elapsed time:",time.time() - T0)
# for i in range(len(train_sents)):
#     print( train_feats[i], X_train[i])
# X_train = sp.sparse.bsr_matrix(np.array(X_train))
# TODO: play with other models
model = LogisticRegression()
# model = Perceptron(); #model.fit(X_train, train_labels)# verbose=1# Whether to print progress messages to stdoutmodel.fit(X_train, train_labels)
# model = LinearSVC(penalty = "l1");# not work
# model = GaussianNB(); # not work
# model = AdaBoostClassifier() # F1-score 20
# model = GradientBoostingClassifier() # too slow
model.fit(X_train, train_labels)
del X_train
print("Elapsed time:",time.time() - T0)


# In[4]:


dev_sents = list(conll2002.iob_sents('esp.testa'))
test_feats = []
test_labels = []

# with open("content_dev.txt",'w')  as f:
#     for sent in dev_sents:
#         f.write(" ".join([ele[0] for ele in sent]) + '\n')


T0 = time.time()
for sent in dev_sents:
    test_feats+=[word2features(sent,i) for i in range(len(sent))]
    test_labels+=[sent[i][-1] for i in range(len(sent))]
print("Elapsed time:",time.time() - T0)

X_test = vectorizer.transform(test_feats)#sparse
# X_test = X_test.toarray()
# scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_test); X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)

j = 0
print("Writing to results.txt")
# format is: word gold pred
with open("results.txt", "w") as out:
    for sent in dev_sents: 
        for i in range(len(sent)):
            word = sent[i][0]
            gold = sent[i][-1]
            pred = y_pred[j]
            j += 1
            out.write("{}\t{}\t{}\n".format(word,gold,pred))
    out.write("\n")

print("Now run: python conlleval.py results.txt")
del test_feats
del test_labels
del dev_sents
del X_test
del y_pred
print("Elapsed time:",time.time() - T0)


# In[5]:


test_sents = list(conll2002.iob_sents('esp.testb'))
#-------------
# with open("content_test.txt",'w')  as f:
#     for sent in test_sents:
#         f.write(" ".join([ele[0] for ele in sent]) + '\n')
#-------------
test2_feats = []
test2_labels = []
T0 = time.time()
for sent in test_sents:
    test2_feats+=[word2features(sent,i) for i in range(len(sent))]
    test2_labels+=[sent[i][-1] for i in range(len(sent))]
print("Elapsed time:",time.time() - T0)

X_test2 = vectorizer.transform(test2_feats)
# X_test2 = X_test2.toarray()
# scaler = sklearn.preprocessing.StandardScaler(with_mean=False); scaler.fit(X_test2); X_test2 = scaler.transform(X_test2)
y_pred2 = model.predict(X_test2)
j=0
print("Writing to results.txt")
# format is: word gold pred
with open("constrained_results.txt", "w") as out:
    for sent in test_sents: 
        for i in range(len(sent)):
            word = sent[i][0]
            gold = sent[i][-1]
            pred = y_pred2[j]
            j += 1
            out.write("{}\t{}\t{}\n".format(word,gold,pred))
    out.write("\n")

print("Now run: python conlleval.py constrained_results.txt")
print("Elapsed time:",time.time() - T0)

