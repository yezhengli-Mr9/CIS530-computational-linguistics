
# coding: utf-8

# In[38]:


import time
T0 = time.time()
test_flag = True
Input_filename = '_input.txt'
if test_flag: Input_filename = 'data/test'+Input_filename #'data/dev_input.txt' # 'data/test_input.txt'#
else: Input_filename = 'data/dev'+Input_filename #'data/dev_input.txt' # 'data/test_input.txt'#
    

Output_filename = '_output_dense.txt'
if test_flag: Output_filename = 'test' + Output_filename #'test_output_features.txt' # 'dev_output_features'
else: Output_filename = 'dev' + Output_filename #'test_output_features.txt' # 'dev_output_features'



# In[39]:


T0 = time.time()
import numpy as np
FeatureMatrix = []
n_words = 0
n_features = 0

with open("/Users/yezheng/Documents/glove.6B/glove.6B.100d_yezheng.txt",'r') as f:
# with open("/Users/yezheng/Documents/coocvec-1000mostfreq-window-3-yezheng.vec",'r') as f:
    n_words, n_features = f.readline().split()
    n_words = int(n_words)
    n_features = int(n_features)
    firstWordLine = f.readline().split()
    WordList = [firstWordLine[0]]
    FeatureMatrix = np.array([list(map(float,firstWordLine[1:] ) ) + [0]] )
#     print(FeatureMatrix.shape) # DEBUG
    for line in f:
        LineSplit = line.split()
        WordList.append(LineSplit[0])
#         print("--------",len(list(map(float,LineSplit[1:]))))
        FeatureMatrix = np.concatenate((FeatureMatrix,np.array([list(map(float,LineSplit[1:])) + [0]] ) )) 
        # last one 0 or 1 means it appears or not
FeatureMatrix = np.concatenate((FeatureMatrix, [np.concatenate((np.zeros(n_features),[1])) ]))
n_features += 1 
#------------------
from gensim.models import KeyedVectors
vecs = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.filter")
n_features2 = len(vecs['picnic'])
print(time.time() - T0)


# In[40]:


T0 = time.time()
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
import random
from operator import itemgetter        
        
# Output_filename = 'test_output_leaderboard.txt'
fd = open(Output_filename,'w') # initialization
fd.close()
with open(Input_filename,'r') as f:
    for line in f:
        LSplit = line.split()
        N_clusters = int(LSplit[2])
        TargetWord = LSplit[0]
        WLstForTarget = LSplit[4:]
#         print(WLstForTarget)
#         print("WordList",WordList)
        IdxSet = [WordList.index(wd) if wd in WordList else -1 for wd in WLstForTarget]
#         print(type(IdxSet))
        Xnew = FeatureMatrix[IdxSet]
        #----------
        Xnew2 = [ np.concatenate((vecs[wd],[0])) if wd in vecs else [0]*n_features2+[1] for wd in WLstForTarget]
        Xnew2 = np.array(Xnew2)
#         print(Xnew.shape,Xnew2.shape)
        Xnew = np.concatenate((Xnew, Xnew2),axis = 1)
        #----------
        scaler = StandardScaler(); scaler.fit(Xnew); Xnew = scaler.transform(Xnew)
        NestedLst = []# reassurance
        NestedLst = defaultdict(list)
        y_pred = [] # reassurance
#         ---------
        ##Kmeans
        kmeans = KMeans(n_clusters= N_clusters).fit(Xnew)        
        for i in range(len(WLstForTarget)): NestedLst[kmeans.labels_[i]].append(WLstForTarget[i])
# # #         -------
# #         #SpectralClustering
# #         NestedLst = []# reassurance
# #         NestedLst = defaultdict(list)
# #         if N_clusters < Xnew.shape[0]: 
# #             y_pred = SpectralClustering(n_clusters=N_clusters).fit_predict(Xnew)
# #         else: y_pred = range(N_clusters) 
#         # ------------
# #         # GMM
        NestedLst = []# reassurance
        NestedLst = defaultdict(list)
        clf = GMM(n_components = N_clusters)
        clf.fit(Xnew)
        y_pred = clf.predict(Xnew)
        
        # avoid empty clusters by Kmeans
        y_pred_missing = set(range(N_clusters)) - set(y_pred)
        while y_pred_missing:
            i = y_pred_missing.pop()
            Kmeans_i_idx = [idx for idx, lbl in enumerate(kmeans.labels_) if i == lbl]
            r_idx = random.randrange(len(Kmeans_i_idx))
            y_pred[Kmeans_i_idx[r_idx]] = i # 
            y_pred_missing = set(range(N_clusters)) - set(y_pred)
        for i in range(len(WLstForTarget)): NestedLst[y_pred[i]].append(WLstForTarget[i])

        with open(Output_filename,'a') as f_output:
            for i in range(N_clusters):
                f_output.write(TargetWord+" :: "+str(i+1)+" ::")
                for w in NestedLst[i]: f_output.write( " " + w)    
                f_output.write('\n')       
print(time.time() - T0)


# In[8]:


T0 = time.time()
import numpy as np
from gensim.models import KeyedVectors
vecs = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.filter")
FeatureMatrix = []
n_features = len(vecs['picnic'])

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

fd = open(Output_filename,'w') # initialization
fd.close()
with open(Input_filename,'r') as f:
    for line in f:
        LSplit = line.split()
        N_clusters = int(LSplit[2])
        TargetWord = LSplit[0]
        WLstForTarget = LSplit[4:]
#         print(WLstForTarget)
#         print("WordList",WordList)
        Xnew = [ np.concatenate((vecs[wd],[0])) if wd in vecs else [0]*n_features+[1] for wd in WLstForTarget]
        Xnew = np.array(Xnew)
        scaler = StandardScaler(); scaler.fit(Xnew); Xnew = scaler.transform(Xnew)
        
        
        NestedLst = []# reassurance
        NestedLst = defaultdict(list)
        y_pred = [] # reassurance
#         ---------
        ##Kmeans
        kmeans = KMeans(n_clusters= N_clusters).fit(Xnew)        
        for i in range(len(WLstForTarget)): NestedLst[kmeans.labels_[i]].append(WLstForTarget[i])
#         -------
        ##SpectralClustering
#         NestedLst = []# reassurance
#         NestedLst = defaultdict(list)

#         if N_clusters < Xnew.shape[0]: 
#             y_pred = SpectralClustering(n_clusters=N_clusters).fit_predict(Xnew)
#         else: y_pred = range(N_clusters)
            
            
            
        # ------------
        # GMM
        NestedLst = []# reassurance
        NestedLst = defaultdict(list)
        clf = GMM(n_components = N_clusters)
        clf.fit(Xnew)
        y_pred = clf.predict(Xnew)
        
        # avoid empty clusters by Kmeans
        y_pred_missing = set(range(N_clusters)) - set(y_pred)
#         if "expect.v"==TargetWord: 
#             print(TargetWord, N_clusters,y_pred_missing)
#             print( kmeans.labels_)
#             print( y_pred)
        while y_pred_missing:
            i = y_pred_missing.pop()
            Kmeans_i_idx = [idx for idx, lbl in enumerate(kmeans.labels_) if i == lbl]
            r_idx = random.randrange(len(Kmeans_i_idx))
            y_pred[Kmeans_i_idx[r_idx]] = i # 
            y_pred_missing = set(range(N_clusters)) - set(y_pred)
#             print(i,r_idx,Kmeans_i_idx)
#             if "expect.v"==TargetWord: 
#                 print(kmeans.labels_)
#                 print(y_pred)
        for i in range(len(WLstForTarget)): NestedLst[y_pred[i]].append(WLstForTarget[i])
#         if "expect.v"==TargetWord: 
#             print(kmeans.labels_)
#             print(y_pred)
#             break
    #----------
        with open(Output_filename,'a') as f_output:
            for i in range(N_clusters):
                f_output.write(TargetWord+" :: "+str(i+1)+" ::")
                for w in NestedLst[i]: f_output.write( " " + w)    
                f_output.write('\n')       
print(time.time() - T0)


# In[11]:


print(Xnew.shape, len(WLstForTarget))


# In[ ]:


# import numpy as np
# from gensim.models import KeyedVectors
# # vecs = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.filter")
# FeatureMatrix = []
# n_features = len(vecs['picnic'])

# from sklearn.cluster import KMeans, SpectralClustering
# from sklearn.preprocessing import StandardScaler
# from collections import defaultdict

# Output_filename = '_output_dense.txt'
# if test_flag: 
#     Output_filename = 'test' + Output_filename # 'test_output_dense.txt' # 'dev_output_dense.txt' 
# else: 
#     Output_filename = 'dev' + Output_filename # 'test_output_dense.txt' # 'dev_output_dense.txt' 
    
# fd = open(Output_filename,'w') # initialization
# fd.close()
# with open(Input_filename,'r') as f:
#     for line in f:
#         LSplit = line.split()
#         N_clusters = int(LSplit[2])
#         TargetWord = LSplit[0]
#         WLstForTarget = LSplit[4:]
# #         print(WLstForTarget)
# #         print("WordList",WordList)
#         Xnew = [ np.concatenate((vecs[wd],[0])) if wd in vecs else [0]*n_features+[1] for wd in WLstForTarget]
#         Xnew = np.array(Xnew)
#         scaler = StandardScaler(); scaler.fit(Xnew); Xnew = scaler.transform(Xnew)
# #         ---------
#         ###Kmeans
#         kmeans = KMeans(n_clusters= N_clusters).fit(Xnew)        
#         for i in range(len(WLstForTarget)): NestedLst[kmeans.labels_[i]].append(WLstForTarget[i])
#         #-------
# #         ###SpectralClustering
# #         y_pred = SpectralClustering(n_clusters=N_clusters).fit_predict(Xnew)
# #         for i in range(len(WLstForTarget)): NestedLst[y_pred[i]].append(WLstForTarget[i])
# #         #-----
#         for i in range(len(WLstForTarget)): NestedLst[kmeans.labels_[i]].append(WLstForTarget[i])
#         with open(Output_filename,'a') as f_output:
#             for i in range(N_clusters):
#                 f_output.write(TargetWord+" :: "+str(i+1)+" ::")
#                 for w in NestedLst[i]: f_output.write( " " + w)    
#                 f_output.write('\n')       
# print(time.time() - T0)


# In[ ]:


# Filter = []
# with open("Vocab.txt",'r') as fd_vocab: Filter = [word[:-1] for word in fd_vocab]
# Filter = set(Filter)
# del FeatureMatrix
# with open("/Users/yezheng/Documents/glove.6B/glove.6B.300d.txt",'r') as f:
#     firstWordLine = f.readline().split()
#     while not firstWordLine[0] in Filter: firstWordLine = f.readline().split()
#     WordList = [firstWordLine[0]]
#     n_features = len(firstWordLine) - 1
# #     print(FeatureMatrix.shape) # DEBUG
#     FeatureMatrix = np.array([list(map(float,LineSplit[1:]))]  )
#     for line in f:
#         LineSplit = line.split()
#         if LineSplit[0] in Filter:
#             WordList.append(LineSplit[0])
#     #         print("--------",len(list(map(float,LineSplit[1:]))))
# #             print(np.array([list(map(float,LineSplit[1:]))]  ).shape)
#             FeatureMatrix = np.concatenate((FeatureMatrix,np.array([list(map(float,LineSplit[1:]))]  ) )) 
#             # last one 0 or 1 means it appears or not
# with open("/Users/yezheng/Documents/glove.6B/glove.6B.300d_yezheng.txt",'w') as out:
#     out.write("{0} {1}\n".format(FeatureMatrix.shape[0], FeatureMatrix.shape[1]))
#     for i,row in enumerate(FeatureMatrix):
#         out.write("{0} {1}\n".format(WordList[i], " ".join(map(lambda s: str(s), FeatureMatrix[i]))))

