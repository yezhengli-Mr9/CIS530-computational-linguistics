
# coding: utf-8

# In[1]:


get_ipython().magic(u'run -i tagger.py --model /Users/yezheng/spanish.005/ --input dataset/esp.testb --output unconstrained_results_tag.txt')


# In[2]:


from nltk.corpus import conll2002
# train_sents = list(conll2002.iob_sents('esp.train'))

# with open("train_sents.txt",'w') as f:
#     for sent in train_sents:  
#         for word_line in sent: f.write(word_line[0]+'\t' +word_line[1]+'\t' +word_line[2]+'\n')
#         f.write('\n')
# dev_sents = list(conll2002.iob_sents('esp.testa'))

# with open("dev_sents.txt",'w') as f:
#     for sent in dev_sents:  
#         for word_line in sent: f.write(word_line[0]+'\t' +word_line[1]+'\t' +word_line[2]+'\n')
#         f.write('\n')
        
# test_sents = list(conll2002.iob_sents('esp.testb'))
# with open("test_sents.txt",'w') as f:
#     for sent in test_sents:          
#         for word_line in sent: f.write(word_line[0]+'\t' +word_line[1]+'\t' +word_line[2]+'\n')
#         f.write('\n')


# In[3]:


test_sents = list(conll2002.iob_sents('esp.testb'))
y_pred2 =[]
with open("unconstrained_results_tag.txt", "r") as f:
    for line in f.readlines():
        line_split = line.split()
        if len(line_split) >0: 
            y_pred2.append(line_split[-1].split("__")[1])
            
print(len(y_pred2))
print(y_pred2)


# In[4]:


j=0
with open("unconstrained_results.txt", "w") as out:
    for sent in test_sents: 
        for i in range(len(sent)):
            word = sent[i][0]
            gold = sent[i][-1]
            pred = y_pred2[j]
            j += 1
            out.write("{}\t{}\t{}\n".format(word,gold,pred))
    out.write("\n")


# In[5]:


get_ipython().magic(u'run -i ../conlleval.py unconstrained_results.txt')

