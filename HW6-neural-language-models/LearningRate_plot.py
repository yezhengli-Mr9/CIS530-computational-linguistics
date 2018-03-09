
# coding: utf-8

# In[3]:


import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

spamReader = csv.reader(open('all_losses.csv', newline=''), delimiter=',', quotechar='|') # too large
# spamReader = csv.reader(open('small_losses.csv', newline=''), delimiter=',', quotechar='|')
row_num = 0
for row in spamReader:
    if row_num in [1,2,4,5, 9]:
    #     print(', '.join(row))
    #     print("row",row)    
    #     for i in row: print(float(i))
        LearningRate_str = row[0]
        row_float = [float(i) for i in row[1:]]
        row_np = np.array(row_float).reshape((2000 , -1))
        row_np = np.mean(row_np, axis = 0)
        x = [i for i in range(len(row_np))]
        plt.plot(x, row_np, label = "$\gamma=$"+LearningRate_str)
    row_num+= 1
plt.ylabel('loss')
plt.xlabel('iteration step of printing epoch')    
plt.title("With various learning rates $\gamma$")
plt.legend(loc='upper right')
plt.show()

