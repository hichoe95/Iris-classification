
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from __future__ import print_function
import os.path
import torch.nn.parallel
from PIL import Image
import PIL
import random
torch.manual_seed(1234)


# In[2]:


f = open('../data/iris.txt', 'r').readlines()

N = len(f) - 1
iris = []

for i in range(0,N):
    line = f[i].split(',')
    iris.append(line)

random.shuffle(iris)

# print(iris)

iris = np.asarray(iris)

feature = iris[:,:-1]
feature = feature.astype(np.float)

label = iris[:,-1]

dic = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}


# In[3]:


# divide train data and test data.
train_len = int(len(feature)*0.7)

train_feature = feature[:train_len]
test_feature = feature[train_len:]

train_label = label[:train_len]
test_label = label[train_len:]
# print(train_feature)
# print(test_label)


# In[4]:


torch.Tensor(train_feature).shape


# In[5]:


# make model

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        
        layers = []
        
        cur_dim = 16
        
        layers.append(nn.Linear(4,cur_dim))
        
        for i in range(0,2):
            layers.append(nn.Linear(cur_dim,cur_dim//2))
            layers.append(nn.ReLU(inplace = True))
            cur_dim = cur_dim//2
        
        # make probability for each class
        layers.append(nn.Linear(cur_dim,3))
#         layers.append(nn.Sigmoid())
        
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.main(x)
        
        return out


# In[6]:


nets = model()

# use crossentropyloss
criterion = nn.CrossEntropyLoss()
# use gradient descent optimizer, and apply lr = 0.1
optimizer = torch.optim.SGD(nets.parameters(), lr = 0.01)


# In[7]:


loss_log = []

for epoch in range(0,200):
    running_loss = 0.
    correct = 0
    for i in range(0,train_len):
        input = train_feature[i,:]
        label = int(dic[train_label[i][:-1]])

        input = torch.FloatTensor(input).view(1,-1)
        label = torch.LongTensor([label])
        
        optimizer.zero_grad()
        #     print(label.shape)
        output = nets(input)
        #     print(output.shape, label.shape)
        loss = criterion(output, label)
        running_loss += loss.data
        loss.backward()
        optimizer.step()

        prediction = torch.max(output.data,1)[1]

        correct += 1 if prediction == label else 0
#         print(prediction, label)
        if i%50 == 0:
            print('epoch : %d , loss : %.6f, accuracy : %.6f' % (epoch+1, loss.data[0], 100 * 1.0 * correct/ (i+1)))
    loss_log.append(running_loss/(1.0*train_len))
    


# In[10]:


correct = 0

epochs = [i for i in range(0,200)]

plt.ylim([0,1])
plt.plot(epochs,loss_log)
plt.show()

for i in range(0, len(feature) - train_len):
    input = test_feature[i]
    label = int(dic[test_label[i][:-1]])

    
    input= torch.FloatTensor(input).view(1,-1)
    
    output = nets(input)
    
    prediction = torch.max(output.data,1)[1]
    
    correct += 1 if prediction == label else 0
    
    print('prediction : {} real label : {}'.format(prediction.item(), label))
    print('accuracy : %.4f' % (100 * 1.0 * correct/(i+1)))



# In[ ]:




