#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt


# In[2]:


train_df = pd.read_csv('train.csv', sep='\t')
test_df = pd.read_csv('test.csv', sep='\t')
sample_submission_df = pd.read_csv('sample_submission.csv')


# In[3]:


x_train = train_df.drop(columns = ['label'])
print(x_train)
y_train = train_df['label']
y_train = y_train.replace('label', 0, regex=True)

x_test = test_df.drop(columns = ['id'])
y_test = sample_submission_df['label']
y_true = y_test


# In[4]:


# 建立Token
token = Tokenizer(num_words=3800) #使用Tokenizer模組建立token，建立一個3800字的字典
#讀取所有訓練資料影評，依照每個英文字在訓練資料出現的次數進行排序，前3800名的英文單字會加進字典中
token.fit_on_texts(x_train['text']) 
print(token.word_index) #可以看到它將英文字轉為數字的結果，例如:the轉換成1
#透過texts_to_sequences可以將訓練和測試集資料中的影評文字轉換為數字list
x_train_seq = token.texts_to_sequences(x_train['text'])
x_test_seq = token.texts_to_sequences(x_test['text']) 
print(x_train_seq)
print(x_test_seq)


# In[5]:


# 每一篇影評文字字數不固定，但後續進行深度學習模型訓練時長度必須固定
# 截長補短
x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
x_test = sequence.pad_sequences(x_test_seq, maxlen=380)
#長度小於380的，前面的數字補0 #長度大於380的，截去前面的數字
#變成25000*380的矩陣 = 25000則評論，每則包含380個數字
print(x_train)
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
print(x_train)


# RNN

# In[6]:


modelRNN = Sequential()
modelRNN.add(Embedding(output_dim=32, #輸出的維度是32，希望將數字list轉換為32維度的向量
                        input_dim=3800, #輸入的維度是3800，也就是我們之前建立的字典是3800字 
                        input_length=380)) #數字list截長補短後都是380個數字


# 建立RNN層，建立16個神經元的RNN層
modelRNN.add(SimpleRNN(units=16))
# 建立隱藏層，建立256個神經元的隱藏層，ReLU激活函數
modelRNN.add(Dense(units=256,activation='relu'))
#隨機在神經網路中放棄70%的神經元，避免overfitting
modelRNN.add(Dropout(0.7)) 
# 建立輸出層，Sigmoid激活函數
modelRNN.add(Dense(units=1,activation='sigmoid')) #建立一個神經元的輸出層
modelRNN.summary()


# In[7]:


# 定義訓練模型
modelRNN.compile(loss='binary_crossentropy',
                 optimizer='adam', 
                 metrics=['accuracy'])


# In[8]:


#Loss function使用Cross entropy 
#adam最優化方法可以更快收斂
train_history = modelRNN.fit(x_train,
                             y_train, 
                             epochs=10,
                             batch_size=100, 
                             verbose=2, 
                             validation_split=0.2)


# In[9]:


scores = modelRNN.evaluate(x_test, y_true,verbose=1)
print(scores[1])


# In[10]:


def show_train_history(train, val, accuracy_or_loss):
    # accuracy_or_loss : input 'Accuracy' or 'loss'
    plt.figure()
    plt.plot(train_history.history[train]) 
    plt.plot(train_history.history[val])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel(accuracy_or_loss)
    plt.legend(["train", "validation"], loc="upper left") 
    plt.show()


# In[11]:


show_train_history('accuracy', 'val_accuracy', 'Accuracy')


# In[12]:


show_train_history('loss', 'val_loss', 'Loss')


# LSTM

# In[13]:


modelLSTM = Sequential() #建立模型
modelLSTM .add(Embedding(output_dim=32, #輸出的維度是32，希望將數字list轉換為32維度的向量
                         input_dim=3800, #輸入的維度是3800，也就是我們之前建立的字典是3800字
                         input_length=380)) #數字list截長補短後都是380個數字

# 建立LSTM層 
modelLSTM .add(LSTM(32)) #建立32個神經元的LSTM層
# 建立隱藏層
modelLSTM .add(Dense(units=256,activation='relu')) #建立256個神經元的隱藏層
modelLSTM .add(Dropout(0.7))
# 建立輸出層，建立一個神經元的輸出層
modelLSTM .add(Dense(units=1,activation='sigmoid'))
# 查看模型摘要
modelLSTM .summary()


# In[14]:


modelLSTM.compile(loss='binary_crossentropy',
                 optimizer='adam', 
                 metrics=['accuracy'])
#Loss function使用Cross entropy 
#adam最優化方法可以更快收斂
train_history = modelLSTM.fit(x_train,
                             y_train, 
                             epochs=10,
                             batch_size=100, 
                             verbose=2, 
                             validation_split=0.2)


# In[15]:


scores = modelLSTM.evaluate(x_test, y_true, verbose=1)
print(scores[1])


# In[16]:


show_train_history('accuracy', 'val_accuracy', 'Accuracy')


# In[17]:


show_train_history('loss', 'val_loss', 'Loss')


# In[ ]:




