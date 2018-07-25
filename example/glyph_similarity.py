"""
Usage:
  $ python glyph_similarity.py "a"
"""

import sys

from gensim.models import KeyedVectors

# model = KeyedVectors.load_word2vec_format("../data/convolutional_AE_300.txt",  binary=False)
model = KeyedVectors.load_word2vec_format("../data/convolutional_AE_300.bin", binary=True)

print(model.most_similar(sys.argv[1]))
