
# coding: utf-8

# In[1]:


from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
from itertools import groupby
import multiprocessing
num_cores = multiprocessing.cpu_count()
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from sklearn import linear_model

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, o, tag, freq):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    #print('word',word)
    o = str(o)
    shape_feature = ['X' if ch.isupper() else ch for ch in word]
#     all_upper_FLAG = 1- int(shape_feature == word)
    shape_feature = "".join(['x' if ch.islower() else ch for ch in shape_feature])
    short_shape_feature = "".join([x[0] for x in groupby(shape_feature)]) 
    features = [   (o + 'word', word)
         ,(o + 'word_shape', shape_feature)
         ,(o + 'word_short_shape', short_shape_feature)
         ,(o + 'word_prefix', word[0])
#          (o + "PoS", part_of_speech),
#          (o + "upper",# len(re.findall(r'[A-Z]',word))
#          all_upper_FLAG),
         ,(o + "hyphen", int('-' in word ))
        ,(o + "len_word", len(word))
         ,(o + "last_two", word[-2:])
    ]

    if o ==0:
        features.append((o + "sentence_frequency", freq))
    return features

def getfeats_tag(word, o, tag): 
    #if just word 61.08 in F1-score (window size = 2,3; optimal window size = 2)
    #if just word_prefix 36.54 in F1-score (window size = 2, 10,20; optimal window size = 20)
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    shape_feature = ['X' if ch.isupper() else ch for ch in word]
    all_upper_FLAG = 1- int(shape_feature == word)
#     shape_feature = "".join(['x' if ch.islower() else ch for ch in shape_feature]) #if just shape_feature 36.19 in F1-score (best amongst window size = 2,3,5,10; optimal window size = 10)
#     print(shape_feature)
#     short_shape_feature = "".join([x[0] for x in groupby(shape_feature)]) # if just short_shape_feature 29.43 in F1-score (window size = 2,5,10; optimal window size = 5)
    features = [ (o+"PoS", tag) #yezheng: I like tags
#           ,(o + 'word_shape', shape_feature),
#          ,(o + 'word_short_shape', short_shape_feature)
#          ,(o + 'word_prefix', word[0])
# #          (o + "PoS", part_of_speech)
         ,(o + "upper",# len(re.findall(r'[A-Z]',word))
         all_upper_FLAG)
#          ,(o + "hyphen", int('-' in word ))
#         ,(o + "len_word", len(word))
#          ,(o + "last_two", word[-2:])
    ]
    return features


    

def word2features(sent, i, full_text):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token

    word = sent[i][0]

    sentence = []
    for item in sent:
        word = item[0]
        sentence.append(word)
    sent_frequency = sentence.count(word)
    #sentence_frequency = full_text.count(word)

    win_size = 4 # results2 results3
    win_size_tag = 10
    for o in range(-win_size,win_size+1):
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            tag = sent[i+o][1]
            featlist = getfeats(word, o, tag, sent_frequency)
            features.extend(featlist)
    for o in range(-win_size_tag,win_size_tag + 1):
        if i+o >= 0 and i+o < len(sent):
            featlist = getfeats_tag(sent[i+o][0], o, sent[i+o][1])
            features.extend(featlist)
    return dict(features)


# In[2]:


import time

# if __name__ == "__main__":
# Load the training data
train_sents = list(conll2002.iob_sents('esp.train'))
# dev_sents = list(conll2002.iob_sents('esp.testa'))
# test_sents = list(conll2002.iob_sents('esp.testb'))

full_text_train = []

for sent in train_sents:
    for word in sent:
        full_text_train.append(word[0])

print('done with full text train generation')
T0 = time.time()
train_feats = []
train_labels = []

# for sent in train_sents:
#     #print(sent)
#     #time.sleep(4)

#     for i in range(len(sent)):

#         #print('sent',sent)

#         feats = word2features(sent,i,full_text_train)
#         #print('feats',feats)
#         train_feats.append(feats)
#         train_labels.append(sent[i][-1])
num = len(train_sents)
print(f"We have {num} sentences in all!")
for i_sent in range(num): # faster
    sent = train_sents[i_sent]
    train_feats += [word2features(sent,i,full_text_train) for i in range(len(sent))]
    train_labels += [sent[i][-1] for i in range(len(sent))]
    if 0 == (i_sent+1) % 2000: 
        print("Elapsed time:",time.time() - T0,f"({i_sent/num*100}%)")
print("Elapsed time:",time.time() - T0)
T0 = time.time()   
vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(train_feats)

# TODO: play with other models
#model = Perceptron(verbose=1)

model = linear_model.LogisticRegression( n_jobs =num_cores, verbose = 1)#penalty = 'l1', solver = 'lbfgs',
#model = svm.SVC(C=1.0, kernel='rbf')
#model = RandomForestClassifier(n_estimators=15, max_depth = 80)
#model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, train_labels)
print("Elapsed time:",time.time() - T0)
# switch to test_sents for your final results
#incorporated this to analyze the full text, not just the sentence


# In[3]:


test_sents = list(conll2002.iob_sents('esp.testb'))
test2_feats = []
test2_labels = []
T0 = time.time()
for sent in test_sents:
    test2_feats+=[word2features(sent,i,full_text_train) for i in range(len(sent))]
    test2_labels+=[sent[i][-1] for i in range(len(sent))]
print("Elapsed time:",time.time() - T0)
X_test2 = vectorizer.transform(test2_feats)
# X_test2 = X_test2.toarray()
# scaler = sklearn.preprocessing.StandardScaler(with_mean=False); scaler.fit(X_test2); X_test2 = scaler.transform(X_test2)
y_pred2 = model.predict(X_test2)
j=0
print("Writing to results.txt")
# format is: word gold pred
with open("constrained_results_phase1.txt", "w") as out:
    for sent in test_sents: 
        for i in range(len(sent)):
            word = sent[i][0]
            gold = sent[i][-1]
            pred = y_pred2[j]
            j += 1
            out.write("{}\t{}\t{}\n".format(word,gold,pred))
    out.write("\n")

print("Now run: python conlleval.py constrained_results_phase1.txt")
print("Elapsed time:",time.time() - T0)


# In[8]:


get_ipython().magic(u'run -i conlleval.py constrained_results_phase1.txt')

