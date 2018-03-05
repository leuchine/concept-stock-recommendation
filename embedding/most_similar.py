#!/usr/bin/env python
# -*- coding: utf-8 -*-

from load import *
from gensim.models.word2vec import Word2Vec

#load model
model=Word2Vec.load("vectors")

#concept and embedding
concept="京津冀一体化".lower()
concept_vec=get_word_vec(model, concept)

l=model.wv.most_similar(positive=[concept])
print(l)
