
# coding: utf-8

# In[1]:


from collections import *
from random import random
from random import randrange
import numpy as np
import csv

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

  try: # yezheng -- tackle with "ISO-8859-1"
        data = open(fname, encoding='utf-8', errors='ignore').read()
  except:
        data = open(fname, encoding="ISO-8859-1").read()  # "UTF-8"
  lm = defaultdict(Counter)
  pad = "~" * order # yezheng: this is just setting beginning of a line -- just like <s><s> mentioned in chapter 4
  data = pad + data
  for i in range(len(data)-order):
    history, char = data[i:i+order], data[i+order]
    lm[history][char]+=1
  def normalize(counter): # input is a dictionary
    s = float(sum(counter.values())) + add_k *len(counter)
    return [(c,(cnt+add_k)/s) for c,cnt in counter.items()]
  outlm = {hist:normalize(chars) for hist, chars in lm.items()}
  return outlm

def train_char_lm_2(fname, order=4, add_k=1):

  # TODO: Add your implementation of add-k smoothing.
  #   data = open(fname).read()

  lm = defaultdict(Counter)
  pad = "~" * order # yezheng: this is just setting beginning of a line -- just like <s><s> mentioned in chapter 4

  with open(fname, encoding='utf-8', errors='ignore') as f:

    for row in f.readlines():
      #print('row',row)
      data = pad + row
      for i in range(len(data)-order):
        history, char = data[i:i+order], data[i+order]
        lm[history][char]+=1

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
    
    
def generate_text(lm, order, nletters=500):
  '''Generates a bunch of random text based on the language model.
  
  Inputs:
  lm: The output from calling train_char_lm.
  history: A sequence of previous text.
  order: The length of the n-grams in the language model.
  
  Returns: 
    A letter  
  '''
  history = "~" * order
  out = []
  for i in range(nletters):
    c = generate_letter(lm, history, order)
    history = history[-order:] + c
    out.append(c)
  return "".join(out)

def perplexity(test_filename, lm, order=4):
  '''Computes the perplexity of a text file given the language model.
  Inputs:
    test_filename: path to text file
    lm: The output from calling train_char_lm.
    order: The length of the n-grams in the language model. #yezheng: order can be read from lm?
  '''
  text = open(test_filename, encoding='utf-8', errors='ignore').read()

  if len(list(lm.keys())[0]) != order:
    print(f"order given ({order}) is inconsistent with lm model's order ({len(list(lm.keys())[0])})")
    return -1.0 # normally, return value should not be negative

  pad = "~" * order
  test = pad + text
  # TODO: YOUR CODE HERE
  # Daphne: make sure (num of characters > order)
  logPP = 0

  N = len(test)-order

  for i in range(N):
    history, char = test[i:(i+order)], test[i+order]
    
    if history not in lm: 
      #print('A')
      logPP -= np.log2(16.0/len(lm)) # float("-inf") # yezheng: deal with unknowns

    else:
      dict_temp = dict(lm[history])

      if char not in dict_temp:
        #print('B',char)
        #print(dict_temp)
        logPP -= np.log2(16.0/len(lm)) #float("-inf")  # yezheng: deal with unknowns
      
      else:
        logPP -= np.log2(dict_temp[char])

  logPP = logPP/N


  return logPP # yezheng: should I return it? notice the SPECIFICATION above does not have "Outputs"

def perplexity_test(text, lm, order=4):
  '''Computes the perplexity of a text file given the language model.
  Inputs:
    test_filename: path to text file
    lm: The output from calling train_char_lm.
    order: The length of the n-grams in the language model. #yezheng: order can be read from lm?
  '''

  if len(list(lm.keys())[0]) != order:
    print(f"order given ({order}) is inconsistent with lm model's order ({len(list(lm.keys())[0])})")
    return -1.0 # normally, return value should not be negative

  pad = "~" * order
  test = pad + text
  # TODO: YOUR CODE HERE
  # Daphne: make sure (num of characters > order)
  logPP = 0

  for i in range(len(test)-order):
    history, char = test[i:(i+order)], test[i+order]
    
    if history not in lm: 
      #print('A')
      logPP += np.log2(1.0/len(lm)) # float("-inf") # yezheng: deal with unknowns

    else:
      dict_temp = dict(lm[history])

      if char not in dict_temp:
        #print('B',char)
        #print(dict_temp)
        logPP += np.log2(1.0/len(lm)) #float("-inf")  # yezheng: deal with unknowns
      
      else:
        logPP += np.log2(dict_temp[char])


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
        print(f"length of 'lms' ne length of 'lambdas': {len(lms)}\ne {len(lambdas)}")
        return -1
    ret = 0
    for idx in range(len(lms)):

      ord = len(list(lms[idx].keys())[0])
      short_hist = history[-ord:]
      #print('short hist',short_hist)
      #print('array',lms[idx][short_hist])

      D_temp = dict(lms[idx][short_hist])
      if char in D_temp:
        ret +=  lambdas[idx] *D_temp[char]

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

def set_lambdas1(lms, dev_filename):

  # piazza: @341 as well as @350

  
  vector = [1.0*(i+1) for i in range(len(lms))]
  total = sum(vector)
  vector2 = [i/total for i in vector]

  return vector2

def set_lambdas2(lms, dev_filename):

  # piazza: @341 as well as @350

  
  vector = [1.0*(i+1)*(i+1) for i in range(len(lms))]
  total = sum(vector)
  vector2 = [i/total for i in vector]

  return vector2

def set_lambdas_2(lms, dev_filename):

  # piazza: @341 as well as @350

  
  vector = [1.0*(i+1)*(i+1) for i in range(len(lms))]
  vector = vector[::-1]
  total = sum(vector)
  vector2 = [i/total for i in vector]

  return vector2

'''
        

# if __name__ == '__main__':
print('Training language model')
lm = train_char_lm("shakespeare_input.txt", order=2)
#   print(generate_text(lm, 2))


# In[2]:


lm = train_char_lm("shakespeare_input.txt", order=4)


lm['hell']
'''

# In[3]:


import pprint
import operator

def print_probs(lm, history):
    if not history in lm: 
        print(f"Error: no word '{history}' in lm")
        return -1
    probs = sorted(lm[history],key=lambda x:(-x[1],x[0]))
    pp = pprint.PrettyPrinter()
    pp.pprint(probs)


'''

print_probs(lm, "hell")
print_probs(lm, "shell") # KeyError: 'shell'


# In[4]:


print_probs(lm, "worl")
'''

# In[5]:


from random import random
def generate_letter(lm, history, order):
    history = history[-order:]
    dist = lm[history]
    x = random()
    for c,v in dist:
        x = x - v
        if x <= 0: return c


# In[6]:

'''


lm = train_char_lm("shakespeare_input.txt", order=2)
print(generate_text(lm, 2))


# print(generate_text(lm, 5, 40)) 
lm = train_char_lm("shakespeare_input.txt", order=3)
print(generate_text(lm, 3))

lm = train_char_lm("shakespeare_input.txt", order=4)
print(generate_text(lm, 4))

lm = train_char_lm("shakespeare_input.txt", order=7)
print(generate_text(lm, 7))



# In[7]:

lm = train_char_lm("shakespeare_input.txt", order=5)
print(generate_text(lm, 5, 40))
'''

#INTERPOLATION AND BACKOFF


lm4 = train_char_lm("shakespeare_input.txt", order=4)
lm3 = train_char_lm("shakespeare_input.txt", order=3)
lm2 = train_char_lm("shakespeare_input.txt", order=2)

lms = [lm4,lm3,lm2]
char = 'k'
history = "Thin"

lambdas = set_lambdas(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)


lambdas = set_lambdas1(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)

lambdas = set_lambdas2(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)

lambdas = set_lambdas_2(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)


char = 'e'
history = "Thin"

lambdas = set_lambdas(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)


lambdas = set_lambdas1(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)

lambdas = set_lambdas2(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)

lambdas = set_lambdas_2(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)


char = ' '
history = "Thin"

lambdas = set_lambdas(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)


lambdas = set_lambdas1(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)

lambdas = set_lambdas2(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)

lambdas = set_lambdas_2(lms, "shakespeare_input.txt")
probability = calculate_prob_with_backoff(char, history, lms, lambdas)
print('probability for ',char,' after ',history,' is ',probability)




#PERPLEXITY

'''

lm = train_char_lm("shakespeare_input.txt", order=4)

text = "test_data/nytimes_article.txt"
perp = perplexity(text, lm, order=4)
print('perplexity for ',text,' is ',perp)

text = "test_data/shakespeare_sonnets.txt"
perp = perplexity(text, lm, order=4)
print('perplexity for ',text,' is ',perp)

text = "test_data/amistad_funesta.txt"
perp = perplexity(text, lm, order=4)
print('perplexity for ',text,' is ',perp)


text = "Romeo"
perp = perplexity_test(text, lm, order=4)
print('perplexity for ',text,' is ',perp)

text2 = "Lionel Messi es lo mas grande que hay puto y cagon"
perp = perplexity_test(text2, lm, order=4)
print('perplexity for ',text2,' is ',perp)

text3 = "Tony Blair was Prime Minister"
perp = perplexity_test(text3, lm, order=4)
print('perplexity for ',text3,' is ',perp)

text4 = "Gentlemen, for shame, forbear this outrage"
perp = perplexity_test(text4, lm, order=4)
print('perplexity for ',text4,' is ',perp)

'''
'''

a  = [4,5,6]
print(a[0:])
print(a[1:])
print(a[2:])
print(a[3:])


'''


# In[16]:

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
  # TODO: YOUR CODE HERE
  # Daphne: make sure (num of characters > order)

  logPP = 0

  for i in range(len(data)-order):
    history, char = data[i:(i+order)], data[i+order]
    #print('data',data)
    #print('history',history)
    #print('char',char)
    
    if history not in lm:
      #print('ERROR A')
      #print('history',history)
      logPP += np.log2(1.0/len(lm)) # float("-inf") # yezheng: deal with unknowns

    else:
      dict_temp = dict(lm[history])

      if char not in dict_temp:
        #print('ERROR B')
        #print('char',char)
        #print(dict_temp)
        logPP += np.log2(1.0/len(lm)) #float("-inf")  # yezheng: deal with unknowns
      
      else:
        #print('NO ERROR - C')
        #print('DICT_TEMP[char]',dict_temp[char])
        logPP += np.log2(dict_temp[char])

  return logPP


# Part 3
# All training should be done on the train set and all evaluation 
# (including confusion matrices and accuracy reports) on the validation set. 
# You will need to change the data processing code to get this working. 
# Specifically, you’ll need to modify the code in the 3rd code block to 
# create two variables category_lines_train and category_lines_val. In addition, to handle unicode, you might need to replace calls to open with calls to codecs.open(filename, "r",encoding='utf-8', errors='ignore').
#----------------------
# yezheng: I do not quite understand his specification of "3rd code block"?


'''


order = 2

#train
import os
lms_dict = {}# a dictionary of lms
for filename in os.listdir('train'):
    filepath = 'train/' + filename 
    with open(filepath) as f:
        lms_dict[filename[:2]] = train_char_lm_2(filepath , order=order)
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
            #print('ERROR',cname)
            #print(filename[:2],label_pred)
            err_count +=1
        print(f"{filename}: Error rate {err_count*1.0/len(CityNames)}")




#LEADERBOARD

import os
lms_dict = {}# a dictionary of lms
for filename in os.listdir('train'):
    filepath = 'train/' + filename
    with open(filepath) as f:
        lms_dict[filename[:2]] = train_char_lm_2(filepath , order=order)

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
    #print(label_pred)
    labels.append(label_pred)

thefile = open('labels.txt', 'w')
for label in labels:
  thefile.write("%s\n" % label)


'''






