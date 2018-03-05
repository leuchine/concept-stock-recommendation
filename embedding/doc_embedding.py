from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import os
import collections
import pickle
import gensim
from gensim.models.word2vec import Word2Vec
import numpy as np
import re
import sys

#convert words to numbers
def convert_words_to_number(f, dataset, file_name):
	global common_word
	for l in f:
		words=re.split('\s',l.lower().strip())
		dataset+=[LabeledSentence(words,[file_name])]


location="../"+sys.argv[1]
sentence_location=location+'/sentence/'
print(sentence_location)
dataset=[]

for file in os.listdir(sentence_location):
	if file != '.DS_Store':
		with open(sentence_location+file) as f:
			convert_words_to_number(f, dataset, file)

model = Doc2Vec(min_count=1, window=80, size=300, sample=1e-4, negative=5, workers=8)
model.build_vocab(dataset)
model.train(dataset, total_examples=len(dataset), epochs=1)
print(model["物联网"])
model.save(location+"/vectors")


"""
for i in ids:
    f.write(str(model.docvecs[i].tolist()))
    f.write('\n')
    f2.write(i)
    f2.write('\n')
f.close()
f2.close()
"""