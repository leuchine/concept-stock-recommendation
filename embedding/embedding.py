import os
import collections
import pickle
import gensim
from gensim.models.word2vec import Word2Vec
import numpy as np
import re
import sys

#convert words to numbers
def convert_words_to_number(f, dataset):
	global common_word
	for l in f:
		words=re.split('\s',l.lower().strip())
		dataset+=[words]


location="../"+sys.argv[1]
sentence_location=location+'/sentence/'
print(sentence_location)
dataset=[]

for file in os.listdir(sentence_location):
	if file != '.DS_Store':			
		with open(sentence_location+file) as f:
			convert_words_to_number(f, dataset)

model = Word2Vec(dataset, size=300, window=80, min_count=1, workers=8, iter=3)
print(model["物联网"])

model.save(location+"/vectors")