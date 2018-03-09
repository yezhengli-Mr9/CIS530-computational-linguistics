
# coding: utf-8

# In[1]:


import nltk
import sklearn_crfsuite
import eli5


# In[2]:


from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
from itertools import groupby
import multiprocessing
num_cores = multiprocessing.cpu_count()

import time


# In[3]:


# def getfeats(word, o, tag, freq):
#     """ This takes the word in question and
#     the offset with respect to the instance
#     word """
#     #print('word',word)
#     o = str(o)
#     shape_feature = ['X' if ch.isupper() else ch for ch in word]
# #     all_upper_FLAG = 1- int(shape_feature == word)
#     shape_feature = "".join(['x' if ch.islower() else ch for ch in shape_feature]) #if just shape_feature 36.19 in F1-score (best amongst window size = 2,3,5,10; optimal window size = 10)
#     short_shape_feature = "".join([x[0] for x in groupby(shape_feature)]) # if just short_shape_feature 29.43 in F1-score (window size = 2,5,10; optimal window size = 5)
#     features = [   (o + 'word', word)
#          ,(o + 'word_shape', shape_feature)
#          ,(o + 'word_short_shape', short_shape_feature)
#          ,(o + 'word_prefix', word[0])
# #          (o + "PoS", part_of_speech),
# #          (o + "upper",# len(re.findall(r'[A-Z]',word))
# #          all_upper_FLAG),
#          ,(o + "hyphen", int('-' in word ))
#         ,(o + "len_word", len(word))
#          ,(o + "last_two", word[-2:])
#     ]

#     if o ==0:
#         features.append((o + "sentence_frequency", freq))
#     return features

# def getfeats_tag(word, o, tag): 
#     #if just word 61.08 in F1-score (window size = 2,3; optimal window size = 2)
#     #if just word_prefix 36.54 in F1-score (window size = 2, 10,20; optimal window size = 20)
#     """ This takes the word in question and
#     the offset with respect to the instance
#     word """
#     o = str(o)
#     shape_feature = ['X' if ch.isupper() else ch for ch in word]
#     all_upper_FLAG = 1- int(shape_feature == word)
# #     shape_feature = "".join(['x' if ch.islower() else ch for ch in shape_feature]) #if just shape_feature 36.19 in F1-score (best amongst window size = 2,3,5,10; optimal window size = 10)
# #     print(shape_feature)
# #     short_shape_feature = "".join([x[0] for x in groupby(shape_feature)]) # if just short_shape_feature 29.43 in F1-score (window size = 2,5,10; optimal window size = 5)
#     features = [ (o+"PoS", tag) #yezheng: I like tags
# #           ,(o + 'word_shape', shape_feature),
# #          ,(o + 'word_short_shape', short_shape_feature)
# #          ,(o + 'word_prefix', word[0])
# # #          (o + "PoS", part_of_speech)
#          ,(o + "upper",# len(re.findall(r'[A-Z]',word))
#          all_upper_FLAG)
# #          ,(o + "hyphen", int('-' in word ))
# #         ,(o + "len_word", len(word))
# #          ,(o + "last_two", word[-2:])
#     ]
#     return features


    

# def word2features(sent, i):
#     """ The function generates all features
#     for the word at position i in the
#     sentence."""
#     features = []
#     # the window around the token

#     word = sent[i][0]

#     sentence = []
#     for item in sent:
#         word = item[0]
#         sentence.append(word)
#     sent_frequency = sentence.count(word)
#     #sentence_frequency = full_text.count(word)

#     win_size = 5 # results2 results3
#     win_size_tag = 10
#     for o in range(-win_size,win_size+1):
#         if i+o >= 0 and i+o < len(sent):
#             word = sent[i+o][0]
#             tag = sent[i+o][1]
#             featlist = getfeats(word, o, tag, sent_frequency)
#             features.extend(featlist)
#     for o in range(-win_size_tag,win_size_tag + 1):
#         if i+o >= 0 and i+o < len(sent):
#             featlist = getfeats_tag(sent[i+o][0], o, sent[i+o][1])
#             features.extend(featlist)
#     return dict(features)


# In[4]:




def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

#     features = {
#         'bias': 1.0,
#         'word.lower()': word.lower(),
#         'word[-3:]': word[-3:],
#         'word.isupper()': word.isupper(),
#         'word.istitle()': word.istitle(),
#         'word.isdigit()': word.isdigit(),
#         'postag': postag,
#         'postag[:2]': postag[:2],
#     }
    features = {}
#     if i > 0:
#         word1 = sent[i-1][0]
#         postag1 = sent[i-1][1]
#         features.update({
#             '-1:word.lower()': word1.lower(),
#             '-1:word.istitle()': word1.istitle(),
#             '-1:word.isupper()': word1.isupper(),
#             '-1:postag': postag1,
#             '-1:postag[:2]': postag1[:2],
#         })
#     else:
#         features['BOS'] = True

#     if i < len(sent)-1:
#         word1 = sent[i+1][0]
#         postag1 = sent[i+1][1]
#         features.update({
#             '+1:word.lower()': word1.lower(),
#             '+1:word.istitle()': word1.istitle(),
#             '+1:word.isupper()': word1.isupper(),
#             '+1:postag': postag1,
#             '+1:postag[:2]': postag1[:2],
#         })
    win_size = 4 # results2 results3
    for o in range(-win_size,win_size+1):
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            postag = sent[i+o][1]
            shape_feature = ['X' if ch.isupper() else ch for ch in word]
            all_upper_FLAG = 1- int(shape_feature == word)
            shape_feature = "".join(['x' if ch.islower() else ch for ch in shape_feature])
            short_shape_feature = "".join([x[0] for x in groupby(shape_feature)])
            o_str = str(o)
            features.update({
            o_str+'word.lower()': word,
            o_str+'shape':shape_feature,
            o_str+'short_shape':short_shape_feature,
            o_str+'word_prefix': word[0],
            o_str+'word.isupper': all_upper_FLAG,
            o_str+'postag': postag,
            o_str+'postag[:2]': postag[:2],
            o_str+'-': int('-' in word),
#             o_str+'len': len(word)
        })
    if i == len(sent)-1:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# In[5]:


T0 = time.time()
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
print("Elapsed time:",time.time() - T0)


# In[6]:


T0 =time.time()
model = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=200,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
)
model.fit(X_train, y_train)
print("Elapsed time:",time.time() - T0)


# In[7]:



T0 = time.time()

test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
test2_feats = [sent2features(s) for s in test_sents]
test2_labels = [sent2labels(s) for s in test_sents]
T0 = time.time()
print("Elapsed time:",time.time() - T0)
# X_test2 = vectorizer.transform(test2_feats)
# X_test2 = X_test2.toarray()
# scaler = sklearn.preprocessing.StandardScaler(with_mean=False); scaler.fit(X_test2); X_test2 = scaler.transform(X_test2)
y_pred1 = model.predict(test2_feats)
# print("y_pred2", y_pred2)
import itertools
y_pred2 = list(itertools.chain(*y_pred1))
j=0
print("Writing to results.txt")
# format is: word gold pred
with open("unconstrained_results.txt", "w") as out:
    for sent in test_sents: 
        for i in range(len(sent)):
            word = sent[i][0]
            gold = sent[i][-1]
            pred = y_pred2[j]
            j += 1
            out.write("{}\t{}\t{}\n".format(word,gold,pred))
    out.write("\n")

print("Now run: python conlleval.py unconstrained_results.txt")
print("Elapsed time:",time.time() - T0)

