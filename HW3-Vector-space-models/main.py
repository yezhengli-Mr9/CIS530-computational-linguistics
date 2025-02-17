
# coding: utf-8

# In[1]:


import os, csv, subprocess, re, random
# os.chdir("lyz")
import numpy as np
import time
import multiprocessing, platform


# In[2]:


def read_in_shakespeare():
    '''Reads in the Shakespeare dataset processesit into a list of tuples.
     Also reads in the vocab and play name lists from files.

    Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.

    Returns:
        tuples: A list of tuples in the above format.
        document_names: A list of the plays present in the corpus.
        vocab: A list of all tokens in the vocabulary.
    '''

    tuples = []
    with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
            line_tokens = [token.lower() for token in line_tokens]
            tuples.append((play_name, line_tokens))
        f.close()
    with open('vocab.txt') as f: 
        vocab = [line.strip() for line in f.readlines()]
        f.close()
    with open('play_names.txt') as f: 
        document_names =  [line.strip() for line in f]
        f.close()
    return tuples, document_names, vocab

def create_term_document_matrix(line_tuples, document_names, vocab):
    '''Returns a numpy array containing the term document matrix for the input lines.
    Inputs:
    line_tuples: A list of tuples, containing the name of the document and 
    a tokenized line from that document.
    document_names: A list of the document names
    vocab: A list of the tokens in the vocabulary
    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let m = len(vocab) and n = len(document_names).

    Returns:
        td_matrix: A mxn numpy array where the number of rows is the number of words
          and each column corresponds to a document. A_ij contains the
          frequency with which word i occurs in document j.
    '''
#     from collections import Counter, defaultdict
#     Dict_doc_words_Counter = defaultdict(Counter)
#     for d, wList in line_tuples: Dict_doc_words_Counter[d] += Counter(wList)
    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    docname_to_id = dict(zip(document_names, range(0, len(document_names))))
    n_docs = len(document_names)
    n_words = len(vocab)
    ret = np.zeros((n_words, n_docs), dtype = np.int32)
    for d, wList in line_tuples: 
        doc_idx = docname_to_id[d]
        for w in wList: ret[vocab_to_id[w]][doc_idx] += 1
# --------------
# not parallel
#     ret = [[Dict_doc_words_Counter[doc][w] for w in vocab] for doc in document_names] # fastest on Macbook
# not parallel
# --------------
# parallel
#     num_cores = multiprocessing.cpu_count()
#     print("num of cores:", num_cores)
#     pool = multiprocessing.Pool(processes=num_cores)
#     ret = pool.map(process_impv, ((Dict_doc_words_Counter[doc],vocab) for doc in document_names) ) 
# parallel
# --------------
    return ret


# In[3]:


def compute_cosine_similarity(v1, v2): 
    '''Computes the cosine similarity of the two input vectors.
    Inputs:()
    v1: A nx1 numpy array 
    v2: A nx1 numpy array 

    Returns:
    A scalar similarity value. # a numpy array if multiple dimension
    '''
    ret = sum(np.multiply(v1, v2))
    if 0 == ret: return ret
    ret = ret/ (np.linalg.norm(v1)*np.linalg.norm(v2))
#     if 1 == len(vector1.shape) and 1 == len(vector1.shape): return ret[0] # np.double
    return ret

def compute_jaccard_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
  '''  
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html
    ret = np.sum(np.minimum(vector1,vector2))/(np.sum(np.maximum(vector1, vector2)))
    return ret

def compute_dice_similarity(vector1, vector2):
    '''Computes the cosine similarity of the two input vectors.

  Inputs:
    vector1: A nx1 numpy array
    vector2: A nx1 numpy array

  Returns:
    A scalar similarity value.
    '''
    ret = np.sum(np.minimum(vector1,vector2))/(np.sum(vector1) + np.sum(vector2))
    return ret


# In[ ]:


'''----------------------------------------
The fourth column of will_play_text.csv contains the name of the character who spokeeach line. Using the methods described above, which characters are most similar? Least similar?
----------------------------------------
'''
def read_character_in_shakspeare():
    '''Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.
    Returns:
        tuples: A list of tuples in the above format.
        ch_names: A list of the plays present in the corpus.
        vocab: A list of all tokens in the vocabulary.
    '''

    tuples = []
    with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        ch_names = set()
        for row in csv_reader:
            ch_name = row[4]
            ch_names.add(ch_name)
            line = row[5]
            line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
            line_tokens = [token.lower() for token in line_tokens]
            tuples.append((ch_name, line_tokens))
        f.close()
#     with open('vocab.txt') as f: vocab = [line.strip() for line in f.readlines()]
    ch_names = list(ch_names)
    return tuples, ch_names#, vocab

# def process_impv_rk_chs(args):
#     sim_fn,v1,v2 = args
#     return sim_fn(v1,v2)
def rank_ch_return_max_min_maxIdx_minIdx(target_ch_idx, matrix, sim_fn):
    ''' Ranks the similarity of all of the words to the target word.
  # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:51 PM.
  Inputs:
    target_word_index: The index of the word we want to compare all others against.
    matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.
    similarity_fn: Function that should be used to compared vectors for two word
      ebeddings. Either compute_dice_similarity, compute_jaccard_similarity, or
      compute_cosine_similarity.

  Returns:
    A length-n list of integer word indices, ordered by decreasing similarity to the 
    target word indexed by word_index
  '''
    n_chs = matrix.shape[1]
    v1 = matrix[:,target_ch_idx]   
    SimLst1 = [sim_fn(v1,matrix[:,i]) for i in range(target_ch_idx)]
    SimLst2 = [sim_fn(v1,matrix[:,i]) for i in range((target_ch_idx+1),n_chs)]
#     #--------
#     # parallel
#     num_cores = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(processes=num_cores)
#     SimLst1 = pool.map(process_impv_rk_chs, [(sim_fn,v1,matrix[:,i]) for i in range(target_ch_idx)])    
#     SimLst2 = pool.map(process_impv_rk_chs, [(sim_fn,v1,matrix[:,i]) for i in range((target_ch_idx+1),n_chs)])
#     # parallel
#     #--------
    SimLst = SimLst1 +[0] + SimLst2
    retMaxIdx = np.argmax(SimLst); retMax = SimLst[retMaxIdx]
    SimLst = SimLst1 +[retMax] + SimLst2
    retMinIdx = np.argmin(SimLst); retMin = SimLst[retMinIdx]
    ret = [retMaxIdx, retMax, retMinIdx, retMin]
    return ret


# In[ ]:


tuples, document_names, vocab = read_in_shakespeare()
N= len(vocab)
vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
def read_in_shakespeare_character_sentence():
    character_word={}
    with open('will_play_text.csv') as f:
        csv_reader = csv.reader(f, delimiter=';')
        for row in csv_reader:    
            character = row[4]
            line = row[5]
            line_tokens = re.sub(r'[^a-zA-Z0-9\s]', ' ', line).split()
            line_tokens = [token.lower() for token in line_tokens]
            if(not  character in character_word):
                character_word[character]=np.zeros((N, 1), dtype = np.int32)
            for token in line_tokens: character_word[character][  vocab_to_id[token]  ]+=1
    return character_word

def compute_similarity(character_word, sim_fn):
    character = list(character_word.keys())
    num_character=len(character_word)
    similarity = {}
    for i in range(num_character):
        #print(i) 
        for j in range(i+1, num_character): 
            similarity[(character[i], character[j])]=sim_fn(  character_word[character[i]], character_word[character[j]]   )
    return similarity
    
start_time = time.time()
character_word=read_in_shakespeare_character_sentence()
similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
for sim_fn in similarity_fns:
    similarity_pair = compute_similarity(character_word, sim_fn)  
    elapsed_time = time.time() - start_time
    rank = sorted(similarity_pair, key=similarity_pair.get, reverse=True) 
    print(elapsed_time) 
    print('The most similar character pair is %s, and the  most dissimilar character pair is %s:' % (rank[0], rank[-1]))


# In[ ]:


A = np.zeros((3,3))
print(A[1,:])
print(A[1])


# In[ ]:


# import time,datetime
# #-------------------------
# T0 = time.time()
# print("HW3 report: Mingyang Liu, Yezheng Li",datetime.datetime.now())
# print("------------------------------------PART I--------------------------------------")
# print("------------------------------------PART II-------------------------------------")
# #---------------------------
# # from skeleton_hw3.py
# tuples, document_names, vocab = read_in_shakespeare()
# # print('Computing term document matrix...')
# td_matrix = create_term_document_matrix(tuples, document_names, vocab)
# td_matrix_just_sentence_len = create_term_document_matrix_just_sentence_len(tuples, document_names, vocab)
# print('''\033[1;31;10m----------------------------------------
# The fourth column of will_play_text.csv contains the name of the character who spokeeach line. Using the methods described above, which characters are most similar? Least similar?
# ----------------------------------------\033[0m
# ''')
# tuples, ch_names = read_character_in_shakspeare() #, vocab 
# n_chs = len(ch_names)
# # print('Computing term document matrix... (character)')
# term_ch_matrix = create_term_document_matrix(tuples, ch_names, vocab)
# T1 = time.time()
# # print("Time elapsed:",T1  - T0,"(",T1-T0,")")
# similarity_fns = [compute_cosine_similarity, compute_jaccard_similarity, compute_dice_similarity]
# # Dice's distance violates triangular inequality while first two obey. The frist two can find max pair by three (possibly only two) steps.
# print('''Notice there are 
# -> six lines with empty character '   '  in will_play_text.csv,
# -> not all characters are gender-identifiable, for example, ''
# we discuss four cases (including/ excluding empty character, with all data, focus only on gender-identifiable characters:''')
# print("\033[1;32;10mThe most/least similar pair (including '  ' character)\033[0m")
# # for sim_fn in similarity_fns[:-1]:
# #     print('\nThe most/least similar pair using %s are:' %  sim_fn.__qualname__) 
# #     retMaxIdx, retMax, retMinIdx, retMin = rank_ch_return_max_min_maxIdx_minIdx(1, term_ch_matrix, sim_fn)
# #     # find max
# #     retMaxIdx1, retMax1, retMinIdx1, retMin1 = rank_ch_return_max_min_maxIdx_minIdx(retMaxIdx, term_ch_matrix, sim_fn)
# #     retMaxIdx2, retMax2, retMinIdx2, retMin2 = rank_ch_return_max_min_maxIdx_minIdx(retMaxIdx1, term_ch_matrix, sim_fn)
# #     # find min
# #     retMaxIdx, retMax, retMinIdx1, retMin1 = rank_ch_return_max_min_maxIdx_minIdx(retMinIdx, term_ch_matrix, sim_fn)
# #     retMaxIdx, retMax, retMinIdx2, retMin2 = rank_ch_return_max_min_maxIdx_minIdx(retMinIdx1, term_ch_matrix, sim_fn)
# #     print("most similar pair (",ch_names[retMaxIdx1],',', ch_names[retMaxIdx2],") least similar pair (",ch_names[retMinIdx1],',', ch_names[retMinIdx2], ")")
# # #     print("max pair (",retMaxIdx1,',', retMaxIdx2,") min pair (",retMinIdx1,',', retMinIdx2, ")")
# #     # yezheng: there are six " " ch_names -- explaining the result


# for sim_fn in similarity_fns:
#     print('The max/min pair using %s are:' %  sim_fn.__qualname__)
#     retMatrix = [rank_ch_return_max_min_maxIdx_minIdx(i, term_ch_matrix, sim_fn) for i in range(n_chs)]
#     retMatrix = np.matrix(retMatrix)
#     maxLst = retMatrix[1,:]
#     maxIdxLst = retMatrix[2,:]
#     minLst = retMatrix[3,:]
#     minIdxLst = retMatrix[4,:]

#     i1_max = np.argmax(maxLst)
#     i2_max = maxIdxLst[i1_max]
#     i1_min = np.argmin(minLst)
#     i2_min = minIdxLst[i1_min]
#     print("max pair:(",i1_max,i2_max,")","min pair:(",i1_min,i2_min,")")
# print("\033[1;32;10mThe most/least similar pair (excluding '  ' character)\033[0m")
# term_ch_matrix2 = term_ch_matrix[1:,1:]
# # for sim_fn in similarity_fns[:-1]:
# #     print('\nThe most/least similar pair using %s are:' %  sim_fn.__qualname__) 
# #     retMaxIdx, retMax, retMinIdx, retMin = rank_ch_return_max_min_maxIdx_minIdx(1, term_ch_matrix2, sim_fn)
# #     # find max
# #     retMaxIdx1, retMax1, retMinIdx1, retMin1 = rank_ch_return_max_min_maxIdx_minIdx(retMaxIdx, term_ch_matrix2, sim_fn)
# #     retMaxIdx2, retMax2, retMinIdx2, retMin2 = rank_ch_return_max_min_maxIdx_minIdx(retMaxIdx1, term_ch_matrix2, sim_fn)
# #     # find min
# #     retMaxIdx, retMax, retMinIdx1, retMin1 = rank_ch_return_max_min_maxIdx_minIdx(retMinIdx, term_ch_matrix2, sim_fn)
# #     retMaxIdx, retMax, retMinIdx2, retMin2 = rank_ch_return_max_min_maxIdx_minIdx(retMinIdx1, term_ch_matrix2, sim_fn)
# #     print("most similar pair (",ch_names[retMaxIdx1+1],',', ch_names[retMaxIdx2+1],") least similar pair (",ch_names[retMinIdx1+1],',', ch_names[retMinIdx2+1], ")")
# #     print("max pair (",retMaxIdx1,',', retMaxIdx2,") min pair (",retMinIdx1,',', retMinIdx2, ")")
#     # yezheng: there are six " " ch_names -- explaining the result
# #dice distance    
# for sim_fn in similarity_fns:
#     print('The max/min pair using %s are:' %  sim_fn.__qualname__)
#     retMatrix = [rank_ch_return_max_min_maxIdx_minIdx(i, term_ch_matrix, sim_fn) for i in range(n_chs)]
#     retMatrix = np.matrix(retMatrix)
#     maxLst = retMatrix[1,:]
#     maxIdxLst = retMatrix[2,:]
#     minLst = retMatrix[3,:]
#     minIdxLst = retMatrix[4,:]

#     i1_max = np.argmax(maxLst)
#     i2_max = maxIdxLst[i1_max]
#     i1_min = np.argmin(minLst)
#     i2_min = minIdxLst[i1_min]
#     print("max pair:(",i1_max,i2_max,")","min pair:(",i1_min,i2_min,")")

# T2 = time.time()
# # print("Time elapsed:", T2  - T0,"(",T2-T1,")")
# #------------------------------------------------------------------------------------
# print('''\033[1;31;10m----------------------------------------
# Shakespeare’s plays are traditionally classified into comedies, histories, and tragedies. Can you use these vector representations to cluster the plays?
# ----------------------------------------\033[0m''')
# # https://en.wikipedia.org/wiki/Shakespeare%27s_plays
# Comedies = ['The Tempest','Two Gentlemen of Verona','Merry Wives of Windsor','Measure for measure',       \
# 'A Comedy of Errors', 'Much Ado about nothing', "Loves Labours Lost", "A Midsummer nights dream", \
#             "Merchant of Venice", "As you like it", "Taming of the Shrew", "Alls well that ends well",      \
#             "Twelfth Night", "A Winters Tale", "Pericles", "The Two Noble Kinsmen" ]
# Histories = ["King John", "Henry IV", "Henry V", "Henry VI Part 1", "Henry VI Part 2",      \
#         "Henry VI Part 3","Richard II", "Richard III", "Henry VIII", "Edward III"]
# Tragedies = ["Troilus and Cressida", "Coriolanus", "Titus Andronicus", "Romeo and Juliet", "Timon of Athens",     \
#              "Julius Caesar", "macbeth", "Hamlet", "King Lear", "Othello", "Antony and Cleopatra", "Cymbeline"]
# print('''
# ----------------------------------------
# There are", len(Comedies),"Comedies", len(Histories), "Histories", len(Tragedies), "Tragedies according to https://en.wikipedia.org/wiki/Shakespeare%27s_plays:''')
# labels_true = [(0*(doc in Comedies) + 1*(doc in Histories) + 2*(doc in Tragedies) ) for doc in document_names]
# from sklearn.cluster import KMeans, SpectralClustering 
# # ------
# # KMeans has randomness -- unstable?
# kmeans_model = KMeans(n_clusters=3, random_state=1)
# kmeans_model.fit(td_matrix.transpose())
# labels_pred = list(kmeans_model.labels_)
# # print([document_names[i] for i in range(len(labels_true)) if 0 == labels_true[i]])
# print("")
# print('labels_true',labels_true)
# print('labels_pred',labels_pred)
# print("\tComedies\tHistories\tTragedies")
# Percentage =[0,0,0]
# # for label in range(2): 
# # print("0\t",)
# # ------
# # .fit(), .fit_predict()  -- not work, require square matrix as input? (strange)
# # Spec_model = SpectralClustering(n_clusters=3,affinity='precomputed')
# # labels_pred  = Spec_model.fit_predict(term_ch_matrix.transpose())
# # ------
# # GMM # too slow to run in Macbook(neither jupyter nor terminal python3 shell); process killed in biglab
# # from sklearn.mixture import GaussianMixture
# # estimator = GaussianMixture(n_components=3, max_iter=20, random_state=0)
# # estimator.fit(np.transpose(term_ch_matrix))
# # labels_pred = estimator.predict(np.transpose(term_ch_matrix.transpose))
# #------------------------------------------------------------------------------------


# In[ ]:


# print('''\033[1;31;10m----------------------------------------
# Do the vector representations of female characters differ distinguishably from male ones?
# ----------------------------------------\033[0m''')
# # from sklearn.cluster import KMeans
# # kmeans_model = KMeans(n_clusters=2, random_state=1).fit(term_ch_matrix.transpose())
# # labels_pred = kmeans_model.labels_
# # print('labels_pred',labels_pred)
# with open('Male.txt') as f: 
#     Males = [line.split('\t')[0].strip().lower() for line in f.readlines()]
#     f.close()
# with open('Female.txt') as f: 
#     Females = [line.split('\t')[0].strip().lower() for line in f.readlines()]
#     f.close()
# ch_names2 = [ch.lower() for ch in ch_names[1:]]
# Not_identified = [ch for ch in ch_names2 if not ch in Males and not ch in Females]
# print(len(Not_identified))
# print(Not_identified)

