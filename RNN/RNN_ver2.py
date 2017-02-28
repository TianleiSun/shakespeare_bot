
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import preprocess


# In[2]:

words, lastwords, word_map, last_map = preprocess.preprocess_word_to_num("shakespeare.txt")
index_map = {}
for (w,i) in word_map.items():
    index_map[i] = w


# In[3]:

wordlist = []
for i in words:
    for j in reversed(i):
        wordlist.append(j)


# In[4]:

maxlen = 7
step = 1
sentences = []
next_word = []


# In[5]:

for i in range(0, len(wordlist) - maxlen, step):
    sentences.append(wordlist[i: i + maxlen])
    next_word.append(wordlist[i + maxlen])
print('nb sequences:', len(sentences))


# In[21]:

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(index_map)), dtype=np.bool)
y = np.zeros((len(sentences), len(index_map)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t in range(len(sentence)):
        w = sentence[t]
        X[i, t, w] = 1
    y[i, next_word[i]] = 1


# In[22]:

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(index_map))))
model.add(Dropout(0.2))
model.add(Dense(len(index_map)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[23]:

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[37]:

for iteration in range(5):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=10)

    start_index = random.randint(0, len(wordlist) - maxlen - 1)

    for diversity in [0.5, 1.0, 1.5]:
        print()
        print('----- diversity:', diversity)

        generated = []
        sentence = wordlist[start_index: start_index + maxlen]
        generated.append(sentence)
        print('----- Generating with seed:')
        for i in sentence:
            sys.stdout.write(index_map[i] + " ")

        for i in range(130):
            x = np.zeros((1, maxlen, len(index_map)))
            for t in range(len(sentence)):
                w = sentence[t]
                X[i, t, w] = 1

            preds = model.predict(x, verbose=0)[0]
            next_i = sample(preds, diversity)
            next_w = index_map[next_i]

            generated.append(next_w)
            sentence = sentence[1:]
            sentence.append(word_map[next_w]) 

            sys.stdout.write(next_w + " ")
            sys.stdout.flush()
        print()


# In[ ]:



