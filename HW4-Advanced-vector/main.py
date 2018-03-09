
# coding: utf-8

# In[ ]:


# # makecooccurrences.py
# import time 
# import gzip
# from collections import defaultdict
# import numpy as np
# def makecooccurrences(D= 500,window = 3):
#     T0 = time.time()
#     T1= T0

#     # Location of corpus
#     corpus = "/Users/yezheng/Documents/reuters.rcv1.tokenized.gz"

#     # Word frequencies
#     freq = defaultdict(int)

#     # Calculate frequencies
#     i = 0
#     with gzip.open(corpus, "rt", encoding='utf8', errors="ignore") as f:
#         for line in f:
#             sline = line.strip().split()
#             for token in sline:
#                 freq[token] += 1
#             # Uncomment this for testing.
#             #if i > 100000:
#             #    break
#             i += 1
#     total = i

#     # This maps the context words to integer (and vice versa)
#     topD = sorted(freq.items(), key=lambda p: p[1], reverse=True)[:D]
#     topD_map = {}
#     topD_map_reverse = {}
#     for i,tup in enumerate(topD):
#         word,wfreq = tup
#         topD_map[word] = i
#         topD_map_reverse[i] = word

#     # This maps word to integer (and vice versa)
#     vocab_map = {}
#     vocab_map_reverse = {}
#     for i,word in enumerate(freq):
#         vocab_map[word] = i
#         vocab_map_reverse[i] = word

#     # Now build term-context matrix
#     M = np.zeros((len(freq), D))

#     k = 0
#     with gzip.open(corpus, "rt", encoding='utf-8', errors="ignore") as f:
#         for line in f:
#             sline = line.strip().split()

#             for i in range(0,len(sline)):
#                 for j in range(i+1, min(len(sline), i+window+1)):
#                     if i == j: continue

#                     # add (i,j)
#                     token_id = vocab_map[sline[i]]
#                     context_token = sline[j]
#                     if context_token in topD_map:
#                         context_token_id = topD_map[context_token]
#                         M[token_id][context_token_id] += 1

#                     # add (j,i)
#                     token_id = vocab_map[sline[j]]
#                     context_token = sline[i]
#                     if context_token in topD_map:
#                         context_token_id = topD_map[context_token]
#                         M[token_id][context_token_id] += 1

#             k += 1
#             if k%10000 == 0:
#                 print("Progress:", k/float(total),"[Eplased time:",time.time() - T0,'(',time.time() - T1 ,")]" )
#                 T1 = time.time()

#             # Uncomment this for testing.
#             #if k > 10000:
#             #    break

#     # Write out to file.
#     with open("coocvec-{}mostfreq-window-{}.vec".format(D, window), "w") as out:
#         out.write("{0} {1}\n".format(M.shape[0], M.shape[1]))
#         for i,row in enumerate(M):
#             out.write("{0} {1}\n".format(vocab_map_reverse[i], " ".join(map(lambda s: str(s), M[i]))))
#     print("Eplased time:",time.time() - T0)


# In[ ]:


import numpy as np
FeatureMatrix = []
# with open("data/GoogleNews-vectors-negative300.filter",'r') as f:
# with open("data/coocvec-500mostfreq-window-3.vec.filter",'r') as f:
with open("/Users/yezheng/Documents/coocvec-500mostfreq-window-3.vec",'r') as f:
    num_words, D_unknown = f.readline().split()
    num_words = int(num_words)
    D_unknown = int(D_unknown)
    firstWordLine = f.readline().split()
    WordList = [firstWordLine[0]]
    FeatureMatrix = np.array([list(map(float,firstWordLine[1:] ) )] )
    for line in f:
        LineSplit = line.split()
        WordList.append(LineSplit[0])
        FeatureMatrix = np.concatenate((FeatureMatrix,np.array([list(map(float,LineSplit[1:]))] ) ))
X = FeatureMatrix


# In[ ]:


from sklearn.cluster import KMeans
from collections import defaultdict
with open('data/dev_input.txt','r') as f:
    for line in f:
        LSplit = line.split()
        N_clusters = int(LSplit[2])
        TargetWord = LSplit[0]
        WLstForTarget = LSplit[4:]
        print(WLstForTarget)
        print("WordList",WordList)
        IdxSet = [WordList.index(wd) for wd in WLstForTarget]
        Xnew = FeatureMatrix[IdxSet, IdxSet]
        kmeans = KMeans(n_clusters= N_clusters).fit(Xnew)
        NestedLst = []# reassurance
        NestedLst = defaultdict(list())
        for i in range(len(WLstForTarget)): NestedLst[kmeans.labels_[i]].append(WLstForTarget[i])
        with open('data/dev_output_lyz.txt','r') as f_output:
            for i in range(N_clusters):
                f_output.write(TargetWord+" :: "+str(i)+" ::")
                for w in NestedLst[i]: f_output.write( " " + NestedLst)    
                f_output.write('\n')       


# In[ ]:


L = "provide.v :: 7 :: ramp wage computerise yield charge articulate nourish upholster fix match glut rail transistorise furnish crenel edge fire canal engage grate bewhisker cater dish interleave indulge arm sustain reflectorise capitalise alphabetize provision transistorize headquarter fulfil subtitle feed fill date sanitate gratify shelter offer pour fret kern help heat pump fulfill railroad scant computerize regale gutter dado hat qualify purvey serve accommodate equip theme joint ply cloy satisfy headline tool coal oversupply seat staff procure ready uniform calk berth extend cleat shower stock curtain skimp whisker glass border flood tube support stipulate surfeit wive board slat set stint key canalize machicolate causeway costume water signalize bush give patch leave corbel signalise rim pimp render outfit crenelate crenellate gate tap bottom horse terrace victual reflectorize copper-bottom step power drench bed leverage fund brattice index meet air-cool nurture supply hydrate allow glaze condition retrofit capitalize caption top constitutionalize specify fuel ticket terrasse prepare innervate air-condition rafter pander underlay cornice wharf partner canalise fit treat hobnail afford toggle".split()
L.index('7')


# In[ ]:


# from sklearn.cluster import KMeans
# k = 5 # according to http://computational-linguistics-class.org/assignment4.html
# kmeans = KMeans(n_clusters=k).fit(X)
# for i in kmeans.labels_: print(i)

