"""
Usage:
   $ python make_character_embedding.py
"""

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import KeyedVectors

text_list = ["あらゆる現実をすべて自分のほうへねじ曲げたのだ。"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts([list(t) for t in text_list])

# model = KeyedVectors.load_word2vec_format("../data/convolutional_AE_300.txt",  binary=False)
model = KeyedVectors.load_word2vec_format("../data/convolutional_AE_300.bin", binary=True)

word_index = tokenizer.word_index
num_words = len(word_index)

embedding_matrix = np.zeros((num_words + 1, 300))
for word, i in word_index.items():
    if word in model.index2word:
        embedding_matrix[i] = model[word]

print(embedding_matrix)
print(embedding_matrix.shape)
