#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import collections
import pickle
from gensim.models.word2vec import Word2Vec

#for filtering digits
def isdigit(s):
    try:
        float(s)
    except:
        return True

    return False


sources=['../finance','../wiki','../bing'] #
all_words=[]
#add all words
for source in sources:
    for file in os.listdir(source+"/sentence"):
        with open(source+"/sentence/"+file) as f:
            all_words+=list(filter(isdigit,f.read().split()))

#check how many words in total
s=set()
for word in all_words:
    s.add(word)
print(len(s))

#dictionary size
vocab=200000
gap=1
vocab_size=vocab-gap

#take out frequent words 
counter=collections.Counter(all_words)
common_word=dict(counter.most_common(vocab_size))

#number them
start=1
for key in common_word:
    common_word[key]=start
    start+=1
#write out 
pickle.dump(common_word, open('dictionary', 'wb'))

#construct word2vec matrix for reinforcement learning initialization
word_vectors=Word2Vec.load("../finance/vectors")
word2vec=[[0]*300]
for number, word in sorted(zip(common_word.values(), common_word.keys())):
    try:

        word2vec.append(word_vectors.wv[word].tolist())
    except KeyError: 
        print(word+ " not found")
        word2vec.append([0]*300)
pickle.dump(word2vec, open('./word2vec', 'wb'))
print(len(word2vec))

print(word_vectors.wv['北京'])