
# coding: utf-8

# In[ ]:


# makecooccurrences.py
import time 
import gzip
from collections import defaultdict
import numpy as np
def makecooccurrences(D= 500,window = 3):
    T0 = time.time()
    T1= T0
    # Location of corpus
    corpus = "/Users/yezheng/Documents/reuters.rcv1.tokenized.gz"
    Filter = []
    with open("Vocab.txt",'r') as fd_vocab: Filter = [word[:-1] for word in fd_vocab]
    Filter = set(Filter)
    # Word frequencies
    freq = defaultdict(int)
    # Calculate frequencies
    i = 0
    with gzip.open(corpus, "rt", encoding='utf8', errors="ignore") as f:
        for line in f:
            sline = line.strip().split()
            for token in sline:
                freq[token] += 1
            # Uncomment this for testing.
            #if i > 100000:
            #    break
            i += 1
    total = i
    # This maps the context words to integer (and vice versa)
    topD = sorted(freq.items(), key=lambda p: p[1], reverse=True)[:D]
    topD_map = {}
    topD_map_reverse = {}
    for i,tup in enumerate(topD):
        word,wfreq = tup
        topD_map[word] = i
        topD_map_reverse[i] = word

    # This maps word to integer (and vice versa)
    vocab_map = {}
    vocab_map_reverse = {}
    for i,word in enumerate(freq):
        vocab_map[word] = i
        vocab_map_reverse[i] = word
    # Now build term-context matrix
    M = np.zeros((len(freq), D))
    k = 0
    with gzip.open(corpus, "rt", encoding='utf-8', errors="ignore") as f:
        for line in f:
            sline = line.strip().split()
            for i in range(0,len(sline)):
                if not sline[i] in Filter: continue
                print("sline[i]",sline[i],i)
                for j in range(i+1, min(len(sline), i+window+1)):
                    if i == j: continue
                    # add (i,j)
                    token_id = vocab_map[sline[i]]
                    context_token = sline[j]
                    if context_token in topD_map:
                        context_token_id = topD_map[context_token]
                        M[token_id][context_token_id] += 1
                    # add (j,i)
                    token_id = vocab_map[sline[j]]
                    context_token = sline[i]
                    if context_token in topD_map:
                        context_token_id = topD_map[context_token]
                        M[token_id][context_token_id] += 1
            k += 1
            if k%100000 == 0:
                print("Progress:", k/float(total),"[Eplased time:",time.time() - T0,'(',time.time() - T1 ,")]" )
                T1 = time.time()
            # Uncomment this for testing.
            #if k > 10000:
            #    break
    # Write out to file.
    with open("/Users/yezheng/Documents/coocvec-{}mostfreq-window-{}-yezheng.vec".format(D, window), "w") as out:
        out.write("{0} {1}\n".format(M.shape[0], M.shape[1]))
        for i,row in enumerate(M):
            out.write("{0} {1}\n".format(vocab_map_reverse[i], " ".join(map(lambda s: str(s), M[i]))))
    print("Eplased time:",time.time() - T0)


# In[ ]:


makecooccurrences(500,3)


# In[ ]:


# print(500,4,'----------------')
# makecooccurrences(500,4)
print(500,5,'----------------')
makecooccurrences(500,5)
print(1000,'----------------')
makecooccurrences(1000,3)

