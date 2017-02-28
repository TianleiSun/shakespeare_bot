
# coding: utf-8

# In[7]:

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import matplotlib.pyplot as plt


# In[8]:

text = open('Shakespeare.txt').read().lower()
print('corpus length:', len(text))


# In[ ]:

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# one hot bit implementation
maxlen = 100
step = 2
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# building model
# using 1 layer and dropout
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# helper function to sample an index from a probability array
def sample(w, T=1.0):
    w = np.asarray(w).astype('float64')
    w = np.log(w) / T
    expw = np.exp(w)
    w = expw / np.sum(expw)
    p = np.random.multinomial(1, w, 1)
    return np.argmax(p)

# train the model, one iteration and change the number of epochs
for iteration in range(2):
    print()
    history = model.fit(X, y, batch_size=128, nb_epoch=10)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.5, 0.75, 1.5]:
        print()
        print('Temperature:', diversity)
        count = 0
        generated = ''
        sys.stdout.write(generated)

        while (count < 14):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            if next_char == "\n": count += 1

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:



