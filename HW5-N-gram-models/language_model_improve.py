
# coding: utf-8

# In[ ]:


from collections import *
from random import random
import numpy as np
# def train_char_lm(fname, order=4, add_k=1):
#   ''' Trains a language model.
#   This code was borrowed from http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

#   Inputs:
#     fname: Path to a text corpus.
#     order: The length of the n-grams.
#     add_k: k value for add-k smoothing. NOT YET IMPLMENTED

#   Returns:
#     A dictionary mapping from n-grams of length n to a list of tuples.
#     Each tuple consists of a possible net character and its probability.
#   '''
#   fnameLst = fname
#   if isinstance(fname, str): fnameLst = [fname]
#   lm = defaultdict(Counter)
#   for fnm in fnameLst:
#       # TODO: Add your implementation of add-k smoothing.
#     #   data = open(fname).read() 
#       try: # yezheng -- tackle with "ISO-8859-1"
#             data = open(fnm, encoding='utf-8', errors='ignore').read()
#       except:
#             data = open(fnm, encoding="ISO-8859-1").read()  # "UTF-8"
#       pad = "~" * order # yezheng: this is just setting beginning of a line -- just like <s><s> mentioned in chapter 4
#       data = pad + data
#       AllChars = set(data)
#     #   print("AllChars",AllChars)
#       for i in range(len(data)-order):
#         history, char = data[i:i+order], data[i+order]
#         lm[history][char]+=1
#       del history
#       del char
#       del i
#       for his in lm.keys():
#         for ch in AllChars: lm[his][ch]+=0 
#   def normalize(counter): # input is a dictionary
#     s = float(sum(counter.values()) + add_k *len(counter))
# #     print(len(counter), len(counter.keys() ))
#     return [(c,(cnt+add_k)/s) for c,cnt in counter.items()]
#   outlm = {hist:normalize(chars) for hist, chars in lm.items()}
#   return outlm

# def train_char_lm(fname, order=4, add_k=1):
#   ''' Trains a language model.

#   This code was borrowed from http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

#   Inputs:
#     fname: Path to a text corpus.
#     order: The length of the n-grams.
#     add_k: k value for add-k smoothing. NOT YET IMPLMENTED

#   Returns:
#     A dictionary mapping from n-grams of length n to a list of tuples.
#     Each tuple consists of a possible net character and its probability.
#   '''

#   # TODO: Add your implementation of add-k smoothing.
#   #   data = open(fname).read()

#   try: # yezheng -- tackle with "ISO-8859-1"
#         data = open(fname, encoding='utf-8', errors='ignore').read()
#   except:
#         data = open(fname, encoding="ISO-8859-1").read()  # "UTF-8"
#   AllChars = set(data)
#   lm = defaultdict(Counter)
#   pad = "~" * order # yezheng: this is just setting beginning of a line -- just like <s><s> mentioned in chapter 4
#   data = pad + data
#   for i in range(len(data)-order):
#     history, char = data[i:i+order], data[i+order]
#     lm[history][char]+=1
#   del history
#   del char
#   del i
#   for his in lm.keys():
#     for ch in AllChars: lm[his][ch]+=0 
#   def normalize(counter): # input is a dictionary
#     s = float(sum(counter.values())) + add_k *len(counter)
#     return [(c,(cnt+add_k)/s) for c,cnt in counter.items()]
#   outlm = {hist:normalize(chars) for hist, chars in lm.items()}
#   return outlm


def train_char_lm(fname, order=4, add_k=1):
  ''' Trains a language model.

  This code was borrowed from http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

  Inputs:
    fname: Path to a text corpus.
    order: The length of the n-grams.
    add_k: k value for add-k smoothing. NOT YET IMPLMENTED

  Returns:
    A dictionary mapping from n-grams of length n to a list of tuples.
    Each tuple consists of a possible net character and its probability.
  '''

  # TODO: Add your implementation of add-k smoothing.
  #   data = open(fname).read()
#-------------
  lm = defaultdict(Counter)
  fnameLst = fname
  if isinstance(fname, str): fnameLst = [fname]
  lm = defaultdict(Counter)
#   print(fnameLst)
  for fnm in fnameLst:
      try: # yezheng -- tackle with "ISO-8859-1"
            fd = open(fnm, encoding='utf-8', errors='ignore')
      except:
            fd = open(fnm, encoding="ISO-8859-1")
      AllChars = set()
      for data in fd.readlines():
          data = data.lower()
          AllChars.update(data)
          pad = "~" * order # yezheng: this is just setting beginning of a line -- just like <s><s> mentioned in chapter 4
          data = pad + data
          for i in range(len(data)-order):
            history, char = data[i:i+order], data[i+order]
            lm[history][char]+=1
          del history
          del char
          del i
      for his in lm.keys():
        for ch in AllChars: lm[his][ch]+=0 
      fd.close()
#-------------
  def normalize(counter): # input is a dictionary
    s = float(sum(counter.values())) + add_k *len(counter)
    return [(c,(cnt+add_k)/s) for c,cnt in counter.items()]
  outlm = {hist:normalize(chars) for hist, chars in lm.items()}

  return outlm

def generate_letter(lm, history, order):
  ''' Randomly chooses the next letter using the language model.  
  Inputs:
    lm: The output from calling train_char_lm.
    history: A sequence of text at least 'order' long.
    order: The length of the n-grams in the language model. 
  Returns: 
    A letter
  '''
  history = history[-order:]
  dist = lm[history]
  x = random()
  for c,v in dist:
    x = x - v
    if x <= 0: return c
    
    
# def generate_text(lm, order, nletters=500):
#   '''Generates a bunch of random text based on the language model.
  
#   Inputs:
#   lm: The output from calling train_char_lm.
#   history: A sequence of previous text.
#   order: The length of the n-grams in the language model.
  
#   Returns: 
#     A letter  
#   '''
#   history = "~" * order
#   out = []
#   for i in range(nletters):
#     c = generate_letter(lm, history, order)
#     history = history[-order:] + c
#     out.append(c)
#   return "".join(out)
#-------------------------
def perplexity(test_filename, lm, order=4):
  '''Computes the perplexity of a text file given the language model.
  Inputs:
    test_filename: path to text file
    lm: The output from calling train_char_lm.
    order: The length of the n-grams in the language model. #yezheng: order can be read from lm?
  '''
  test = open(test_filename).read()
  if len(lm.keys()[0]) == order: 
    print(f"order given ({order}) is inconsistent with lm model's order ({len(lm.keys()[0])})")
    return -1.0 # normally, return value should not be negative
  pad = "~" * order
  test = pad + test 
  # TODO: YOUR CODE HERE
  # Daphne: make sure (num of characters > order)
  logPP = 0
  for i in range(len(test)-order): 
    history, char = test[i:(i+order)], test[i+order]
    if not history in lm: return -1.0# float("-inf") # yezheng: deal with unknowns
    dict_temp = dict(lm[history])
    if not char in dict_temp: return -1.0 #float("-inf")  # yezheng: deal with unknowns
    logPP += - np.log2(dict_temp[char])
  return logPP # yezheng: should I return it? notice the SPECIFICATION above does not have "Outputs"
    #yezheng: I have not dealt with UNKNOWN/ unseen cases (piazza @335 led a discussion, not quite for sure they resolve this issue)
  



def calculate_prob_with_backoff(char, history, lms, lambdas):
    '''Uses interpolation to compute the probability of char given a series of 
     language models trained with different length n-grams.

   Inputs:
     char: Character to compute the probability of.
     history: A sequence of previous text.
     lms: A list of language models, outputted by calling train_char_lm.
     lambdas: A list of weights for each lambda model. These should sum to 1.
    
  Returns:
    Probability of char appearing next in the sequence.
  ''' 
  # TODO: YOUR CODE HRE
    #yezheng: notice this is interpolation rather than backoff
    #yezheng: Think lambdas are discounting according to page 15 of chapter 4.
    if not len(lms) == len(lambdas): 
        print(f"length of 'lms' \ne length of 'lambdas': {len(lms)}\ne {len(lambdas)}")
        return -1
    ret = 0
    for idx in range(len(lms)):
        ord = len(list(lms[idx].keys())[0])
#         print(history[(len(history)-ord):],"lms[idx][history[(len(history)-ord):]]",lms[idx][history[(len(history)-ord):]])
        D_temp = dict(lms[idx][history[(len(history)-ord):]])
        if char in D_temp: 
            ret +=  lambdas[idx] *D_temp[char]
#         else: return float("-Inf") #yezheng
    return ret
    
    


def set_lambdas(lms, dev_filename):
  '''Returns a list of lambda values that weight the contribution of each n-gram model

  This can either be done heuristically or by using a development set.

  Inputs:
    lms: A list of language models, outputted by calling train_char_lm.
    dev_filename: Path to a development text file to optionally use for tuning the lmabdas. 

  Returns:
    Probability of char appearing next in the sequence. # yezheng: should I return lambdas
  '''
  # TODO: YOUR CODE HERE
  # piazza: @341 as well as @350
  with open(dev_filename) as f:
    Lines = f.readlines()
#    # # EM algorithm
#     for lm in lms:
#         order = lm.keys()[0]
#         pad = "~" * order
#         for line in Lines:
#             data = pad + line
  return [1.0/len(lms)] *len(lms)
        
if __name__ == '__main__':
    print('Training language model')
    lm = train_char_lm("shakespeare_input.txt", order=2)
    lm = train_char_lm("shakespeare_input.txt", order=4)
    lm['hell']
#     print(generate_text(lm, 2))


# In[ ]:


# import pprint
# import operator

def print_probs(lm, history):
    if not history in lm: 
        print(f"Error: no word '{history}' in lm")
        return -1
    probs = sorted(lm[history],key=lambda x:(-x[1],x[0]))
    pp = pprint.PrettyPrinter()
    pp.pprint(probs)
# print_probs(lm, "hell")
# print_probs(lm, "shell") # KeyError: 'shell'
# print_probs(lm, "worl")
from random import random
def generate_letter(lm, history, order):
    history = history[-order:]
    dist = lm[history]
    x = random()
    for c,v in dist:
        x = x - v
        if x <= 0: return c


# In[ ]:


# def perplexity_test(text, lm, order=4):
#   '''Computes the perplexity of a text file given the language model.
#   Inputs:
#     test_filename: path to text file
#     lm: The output from calling train_char_lm.
#     order: The length of the n-grams in the language model. #yezheng: order can be read from lm?
#   '''

#   if len(list(lm.keys())[0]) != order:
#     print(f"order given ({order}) is inconsistent with lm model's order ({len(list(lm.keys())[0])})")
#     return -1.0 # normally, return value should not be negative

#   pad = "~" * order
#   test = pad + text
#   # TODO: YOUR CODE HERE
#   # Daphne: make sure (num of characters > order)
#   logPP = 0

#   for i in range(len(test)-order):
#     history, char = test[i:(i+order)], test[i+order]
    
#     if history not in lm: 
#       #print('A')
#       logPP -= 1.0# float("-inf") # yezheng: deal with unknowns
#     else:
#       dict_temp = dict(lm[history])
#       if char not in dict_temp:
#         #print('B',char)
#         #print(dict_temp)
#         logPP -= 1.0 #float("-inf")  # yezheng: deal with unknowns
        
#       else:
#         logPP -= np.log2(dict_temp[char])
#   return logPP

# def perplexity_yezheng_string(cityname, lm, order=4,filename_debug = ''):
#   '''Computes the perplexity of a text file given the language model.
#   Inputs:
#     test_filename: path to text file
#     lms: The output from calling train_char_lms.
#     order: The length of the n-grams in the language model. #yezheng: order can be read from lm?
#   Outputs:
#     max_labels: a list of predicted labels
#   '''
#   #order = len(list(lm.keys())[0]) #yezheng: I think it should not be an argument
#   pad = "~" * order
#   data = pad + cityname 
#   # TODO: YOUR CODE HERE
#   # Daphne: make sure (num of characters > order)
#   logPP = 0
#   for i in range(len(data)-order):
#     history, char = data[i:(i+order)], data[i+order] 
#     if history not in lm: logPP -= np.log2(len(lm))
#     else:
#       dict_temp = dict(lm[history])
#       if char not in dict_temp:
#         logPP -= np.log2(1.0/len(lm))
#       else:
#         logPP -= np.log2(dict_temp[char])
#   return logPP



def perplexity_yezheng_string(cityname, lm, order=4):
  '''Computes the perplexity of a text file given the language model.
  Inputs:
    test_filename: path to text file
    lms: The output from calling train_char_lms.
    order: The length of the n-grams in the language model. #yezheng: order can be read from lm?
  Outputs:
    max_labels: a list of predicted labels
  '''
  #order = len(list(lm.keys())[0]) #yezheng: I think it should not be an argument
  pad = "~" * order
  data = pad + cityname
  data = data.lower()
  # TODO: YOUR CODE HERE
  # Daphne: make sure (num of characters > order)
  logPP = 0
  for i in range(len(data)-order):
    history, char = data[i:(i+order)], data[i+order]   
    if history not in lm:
      logPP += np.log2(8.0/len(lm)) # float("-inf") # yezheng: deal with unknowns
    else:
      dict_temp = dict(lm[history])
      if char not in dict_temp:
        logPP += np.log2(8.0/len(lm)) #float("-inf")  # yezheng: deal with unknowns
      else:
        logPP += np.log2(dict_temp[char])
  return logPP


# In[ ]:


# Part 3
# All training should be done on the train set and all evaluation 
# (including confusion matrices and accuracy reports) on the validation set. 
# You will need to change the data processing code to get this working. 
# Specifically, you’ll need to modify the code in the 3rd code block to 
# create two variables category_lines_train and category_lines_val. In addition, to handle unicode, you might need to replace calls to open with calls to codecs.open(filename, "r",encoding='utf-8', errors='ignore').
#----------------------
# yezheng: I do not quite understand his specification of "3rd code block"?


order = 4

#train
import os
lms_dict = {}# a dictionary of lms
for filename in os.listdir('train'):
#     filepath = 'train/' + filename 
    filepath = ['train/' + filename,'val/' + filename]
    lms_dict[filename[:2]] = train_char_lm(filepath , order=order)
lms_names = list(lms_dict.keys()) # "af, cn, de, fi, \ldots"

# yezheng: I define this one myself since


# validation set
for filename in os.listdir('val'):
    filepath = 'val/' + filename
    with open(filepath) as f:
        CityNames = f.readlines()
        err_count = 0
        for cname in CityNames:
          PP_lst = [perplexity_yezheng_string(cityname = cname[:-1],lm = lms_dict[CountryName], order=order) for CountryName in lms_names]
          #print('city',cname)
          #print(PP_lst)
          #we need something for cities where all values are -1
          label_pred = lms_names[PP_lst.index(max(PP_lst))]
          #print(label_pred)
          if filename[:2] != label_pred:
            err_count +=1
        print(f"{filename}: Error rate {err_count*1.0/len(CityNames)}")


# In[ ]:


# in.txt: Error rate 0.65
# pk.txt: Error rate 0.39
# fr.txt: Error rate 0.32
# af.txt: Error rate 0.43
# cn.txt: Error rate 0.13
# za.txt: Error rate 0.36
# fi.txt: Error rate 0.32
# ir.txt: Error rate 0.55
# de.txt: Error rate 0.49


# In[ ]:


#LEADERBOARD

order = 2
AddK = 1
import os
lms_dict = {}# a dictionary of lms
for filename in os.listdir('train'):
    filepath = ['train/' + filename,'val/' + filename]
    lms_dict[filename[:2]] = train_char_lm(filepath , order=order, add_k = AddK)

lms_names = list(lms_dict.keys()) # "af, cn, de, fi, \ldots"

#for filename in os.listdir('val'):
#    filepath = 'val/' + filename
#    with open(filepath) as f:
#        lms_dict[filename[:2]] = train_char_lm(filepath , order=order)


print('HERE THEY COME')
filepath2 = 'cities_test.txt'
labels = []
with open(filepath2, encoding='utf-8', errors='ignore') as f:
  CityNames = f.readlines()
  for cname in CityNames:
    PP_lst = [perplexity_yezheng_string(cityname = cname[:-1],lm = lms_dict[CountryName], order=order) for CountryName in lms_names]
    #print('city',cname)
    #print(PP_lst)
    #we need something for cities where all values are -1
    if max(PP_lst) == min(PP_lst):
      label_pred = lms_names[randrange(0,len(lms_names))]
    else:
      label_pred = lms_names[PP_lst.index(max(PP_lst))]
#     print(label_pred)
    labels.append(label_pred)

with open('../labels.txt', 'w') as thefile:
    for label in labels: thefile.write("%s\n" % label)
print("HERE THEY FINISH")


# In[ ]:


# lm4 = train_char_lm("shakespeare_input.txt", order=4)
# lm3 = train_char_lm("shakespeare_input.txt", order=3)
# lm2 = train_char_lm("shakespeare_input.txt", order=2)
# lm1 = train_char_lm("shakespeare_input.txt", order=1)
# lms = [lm4,lm3,lm2]
# # lms = [lm3,lm2,lm1]
# lambdas = set_lambdas(lms, "shakespeare_input.txt")
# hist = "Thin"
# ch = 'e'
# print(calculate_prob_with_backoff(ch,hist, lms, lambdas))

#-------------
# lm = train_char_lm("shakespeare_input2.txt", order=5)
# print(generate_text(lm, 5, 40),"------------")
# print(lm['~~~~~'])
# print(generate_text(lm, 5, 40),"------------")
# print(generate_text(lm, 5, 40),"------------")
# print(generate_text(lm, 5, 40),"------------")
# print(generate_text(lm, 5, 40),"------------")
# lm = train_char_lm("shakespeare_input2.txt", order=4)
# print(generate_text(lm, 4, 40),"------------")
# print(lm['~~~~'])
# print(generate_text(lm, 4, 40),"------------")
# print(generate_text(lm, 4, 40),"------------")
# lm = train_char_lm("shakespeare_input.txt", order=2)
# print(generate_text(lm, 2, 40),"------------")
# print(lm['~~'])
# print(generate_text(lm, 2, 40),"------------")
#--------------
# lm = train_char_lm("shakespeare_input.txt", order=2)
# print(generate_text(lm, 2))
# # print(generate_text(lm, 5, 40)) 

