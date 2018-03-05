#!/usr/bin/env python
# -*- coding: utf-8 -*-

from load import *
from gensim.models.word2vec import Word2Vec
from gensim.models import Doc2Vec
from find_stocks import *

#load stock
stock={}
load_stock(stock)
#load model
model=Doc2Vec.load("../bing2/vectors")

#concept and embedding
concept="环保".lower()
concept_vec=get_word_vec(model, concept)

rank_list=top_stock(concept_vec, stock, model)
print(rank_list[:20])