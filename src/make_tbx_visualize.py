# visualizing character embbeddings by using tensorboardX

import re
from pathlib import Path

import numpy as np
import gensim
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from PIL import Image


vec_path = "data/convolutional_AE_300.bin"
img_path = "data/out/black/NotoSansCJKjp-Regular/"

writer = SummaryWriter()

# load vectors
model = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=True)
weights = model.vectors
labels = model.index2word


features = np.zeros(weights.shape)
images_list = []
for i, img in enumerate(Path(img_path).glob("*.png")):
    img_v = np.array(Image.open(img))
    images_list.append(img_v)

    c = chr(int(re.sub(r"char_([0-9]*).png", "\\1", img.name)))
    index = labels.index(c)
    features[i] = weights[index]

features = torch.FloatTensor(features)
images = torch.FloatTensor(images_list)

# visualize up to 1000 images
features = features[:1000]
images = images[:1000]
writer.add_embedding(features, label_img=images.unsqueeze(1))
