
# coding: utf-8

# In[11]:


import time
T0 = time.time()
test_flag = False
Input_filename = '_input.txt'
if test_flag: Input_filename = 'data/test'+Input_filename #'data/dev_input.txt' # 'data/test_input.txt'#
else: Input_filename = 'data/dev'+Input_filename #'data/dev_input.txt' # 'data/test_input.txt'#
    
Output_filename = '_output_features.txt'
if test_flag: Output_filename = 'test' + Output_filename #'test_output_features.txt' # 'dev_output_features'
else: Output_filename = 'dev' + Output_filename #'test_output_features.txt' # 'dev_output_features'



# In[12]:


import numpy as np
FeatureMatrix = []
T0 = time.time()
n_words = 0
n_features = 0
# with open("data/coocvec-500mostfreq-window-3.vec.filter",'r') as f:
# with open("/Users/yezheng/Documents/coocvec-200mostfreq-window-3-yezheng2.vec",'r') as f:
# with open("/Users/yezheng/Documents/coocvec-500mostfreq-window-5-yezheng2.vec",'r') as f:
# with open("/Users/yezheng/Documents/coocvec-500mostfreq-window-3-yezheng1000.vec",'r') as f:
# with open("/Users/yezheng/Documents/coocvec-500mostfreq-window-4-yezheng2.vec",'r') as f:


# with open("/Users/yezheng/Documents/coocvec-500mostfreq-window-5-yezheng.vec",'r') as f:
# with open("/Users/yezheng/Documents/coocvec-500mostfreq-window-6-yezheng.vec",'r') as f:
# with open("/Users/yezheng/Documents/coocvec-500mostfreq-window-4-yezheng.vec",'r') as f:
# with open("/Users/yezheng/Documents/coocvec-500mostfreq-window-4-yezheng.vec",'r') as f:
# with open("/Users/yezheng/Documents/coocvec-600mostfreq-window-3-yezheng.vec",'r') as f:
with open("/Users/yezheng/Documents/coocvec-1000mostfreq-window-3-yezheng.vec",'r') as f:
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


FeatureMatrix2 = []
n_words2 = 0
n_features2 = 0
with open("/Users/yezheng/Documents/coocvec-500mostfreq-window-4-yezheng.vec",'r') as f2:
    n_words2, n_features2 = f2.readline().split()
    n_words2 = int(n_words2)
    n_features2 = int(n_features2)
    firstWordLine2 = f2.readline().split()
    WordList2 = [firstWordLine2[0]]
    FeatureMatrix2 = np.array([list(map(float,firstWordLine2[1:] ) ) + [0]] )
#     print(FeatureMatrix.shape) # DEBUG
    for line2 in f2:
        LineSplit2 = line2.split()
        WordList2.append(LineSplit2[0])
#         print("--------",len(list(map(float,LineSplit[1:]))))
        FeatureMatrix2 = np.concatenate((FeatureMatrix2,np.array([list(map(float,LineSplit2[1:])) + [0]] ) )) 
        # last one 0 or 1 means it appears or not
FeatureMatrix2 = np.concatenate((FeatureMatrix2, [np.concatenate((np.zeros(n_features2),[1])) ]))
n_features2 += 1 


# In[13]:


T0 = time.time()
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
        
        
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
        Xnew = FeatureMatrix[IdxSet, :]
        #----------
#         IdxSet2 = [WordList2.index(wd) if wd in WordList2 else -1 for wd in WLstForTarget]
#         Xnew2= FeatureMatrix[IdxSet2, :]
#         Xnew = np.concatenate((Xnew, Xnew2),axis = 1)
#----------

        scaler = StandardScaler(); scaler.fit(Xnew); Xnew = scaler.transform(Xnew)
        NestedLst = []# reassurance
        y_pred = [] # reassurance
        NestedLst = defaultdict(list)
#         ---------
        ##Kmeans
        kmeans = KMeans(n_clusters= N_clusters).fit(Xnew)        
        for i in range(len(WLstForTarget)): NestedLst[kmeans.labels_[i]].append(WLstForTarget[i])
#         ---------
            ## SpectralClustering
#         if N_clusters < Xnew.shape[0]: 
#             y_pred = SpectralClustering(n_clusters=N_clusters).fit_predict(Xnew)
#         else: y_pred = range(N_clusters)
# # #         --------
        # GMM
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
#         ---------
        with open(Output_filename,'a') as f_output:
            for i in range(N_clusters):
                f_output.write(TargetWord+" :: "+str(i+1)+" ::")
                for w in NestedLst[i]: f_output.write( " " + w)    
                f_output.write('\n')       
print(time.time() - T0)


# In[5]:


#DEBUG
print(IdxSet)


# In[ ]:


T0 = time.time()
import numpy as np
FeatureMatrix = []
n_words = 0
n_features = 0
# with open("data/GoogleNews-vectors-negative300.filter",'r') as f:
# # with open("data/coocvec-500mostfreq-window-3.vec.filter",'r') as f:
# # with open("/Users/yezheng/Documents/coocvec-500mostfreq-window-4-yezheng.vec",'r') as f:
#     n_words, n_features = f.readline().split()
#     n_words = int(n_words)
#     n_features = int(n_features)
#     firstWordLine = f.readline().split()
#     WordList = [firstWordLine[0]]
#     FeatureMatrix = np.array([list(map(float,firstWordLine[1:] ) ) + [0]] )
# #     print(FeatureMatrix.shape) # DEBUG
#     for line in f:
#         LineSplit = line.split()
#         WordList.append(LineSplit[0])
# #         print("--------",len(list(map(float,LineSplit[1:]))))
#         FeatureMatrix = np.concatenate((FeatureMatrix,np.array([list(map(float,LineSplit[1:])) + [0]] ) )) 
#         # last one 0 or 1 means it appears or not
# FeatureMatrix = np.concatenate((FeatureMatrix, [np.concatenate((np.zeros(n_features),[1])) ]))
# n_features += 1 
# X = FeatureMatrix
# # -----------------------


# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from collections import defaultdict
# Output_filename = '_output_features.txt'
# if test_flag: Output_filename = 'test' + Output_filename #'test_output_features.txt' # 'dev_output_features'
# else: Output_filename = 'dev' + Output_filename #'test_output_features.txt' # 'dev_output_features'
        
        
# # Output_filename = 'test_output_leaderboard.txt'
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
#         IdxSet = [WordList.index(wd) if wd in WordList else -1 for wd in WLstForTarget]
#         Xnew = FeatureMatrix[IdxSet, :]
#         scaler = StandardScaler(); scaler.fit(Xnew); Xnew = scaler.transform(Xnew)

        
#         NestedLst = []# reassurance
#         NestedLst = defaultdict(list)
# #         ---------
#         ##Kmeans
#         kmeans = KMeans(n_clusters= N_clusters).fit(Xnew)        
#         for i in range(len(WLstForTarget)): NestedLst[kmeans.labels_[i]].append(WLstForTarget[i])
# #         -------
#         ##SpectralClustering
#         y_pred = SpectralClustering(n_clusters=N_clusters).fit_predict(Xnew)
#         for i in range(len(WLstForTarget)): NestedLst[y_pred[i]].append(WLstForTarget[i])
#         #-----
#         for i in range(len(WLstForTarget)): NestedLst[kmeans.labels_[i]].append(WLstForTarget[i])
#         with open(Output_filename,'a') as f_output:
#             for i in range(N_clusters):
#                 f_output.write(TargetWord+" :: "+str(i+1)+" ::")
#                 for w in NestedLst[i]: f_output.write( " " + w)    
#                 f_output.write('\n')       
# print(time.time() - T0)


# In[ ]:


# # self attempt
# T0 = time.time()
# import numpy as np
# from gensim.models import KeyedVectors
# vecs = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.filter")
# FeatureMatrix = []
# n_features = len(vecs['picnic'])

# from sklearn.cluster import KMeans
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
# #         Xnew = Xnew[:,:200]
#         scaler = StandardScaler(); scaler.fit(Xnew); Xnew = scaler.transform(Xnew)
#         NestedLst = []# reassurance
#         NestedLst = defaultdict(list)
#         #---------
#         ####Kmeans
# #         kmeans = KMeans(n_clusters= N_clusters).fit(Xnew)        
# #         for i in range(len(WLstForTarget)): NestedLst[kmeans.labels_[i]].append(WLstForTarget[i])
#         #-------
#         ###SpectralClustering
#         y_pred = SpectralClustering(n_clusters=N_clusters).fit_predict(Xnew)
#         for i in range(len(WLstForTarget)): NestedLst[y_pred[i]].append(WLstForTarget[i])
#         #-----
#         with open(Output_filename,'a') as f_output:
#             for i in range(N_clusters):
#                 f_output.write(TargetWord+" :: "+str(i+1)+" ::")
#                 for w in NestedLst[i]: f_output.write( " " + w)    
#                 f_output.write('\n')       
# print(time.time() - T0)

