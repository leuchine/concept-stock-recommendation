#!/usr/bin/env python
# -*- coding: utf-8 -*-

from load import *
from find_stocks import *
from gensim.models.word2vec import Word2Vec
import numpy
import collections
import jieba

def get_vector_representation(query_list, model):
	global ratio
	concept_vec=get_word_vec(model, query_list[0])
	for query in query_list[1:]:
		concept_vec+=ratio*get_word_vec(model,query)
	return concept_vec


#load stock
all_stock={}
load_stock(all_stock)

#load concepts
concepts={}
concept_set=[]
load_concept(concepts, concept_set, 'jinrongjie')
print(concepts)
print(len(concepts))
ssum=0
for i,j in concepts.items():
	ssum+=len(j)
print(ssum)
#load description
description={}
load_description(description)

#for ranking stocks
model=Word2Vec.load("../finance/vectors")
#for exapnsion
model_expansion=Word2Vec.load("../finance/vectors")

#expansion is on or off
expansion=False
#P@5
amount=5
recall_amount=30
ave_pre=0
ave_recall=0
ave_map=0
#ratio
ratio=0.2

split_num=0
for concept in concept_set[split_num:]:
	related_stock=concepts[concept]
	concept=concept.lower()


	#print([concept]+description.get(concept,[]))
	#get representations
	concept_vec=numpy.array(get_vector_representation([concept], model))

	#expand concept
	if expansion:
		vec_for_expansion=numpy.array(get_word_vec(model_expansion, concept))	
		for i in model_expansion.similar_by_vector(vec_for_expansion, topn=8)[1:]:	
			if not isdigit(str(i[0])):
				concept_vec+=ratio*get_word_vec(model,i[0])
	#find stocks
	top=top_stock(concept_vec, all_stock, model)
	print(top[:5])
	#compute precision
	precision=compute_precision(top, amount, related_stock)
	recall=compute_recall(top, recall_amount, related_stock)
	map_score=compute_map(top, related_stock)

	ave_pre+=precision
	ave_recall+=recall
	ave_map+=map_score
	print(concept+" Precision: "+str(precision))
	print(concept+" Recall: "+str(recall))
	print(concept+" MAP: "+str(map_score))
print(len(concepts)-split_num)
print("average precision is: "+str(ave_pre/float(len(concepts)-split_num)))
print("average recall is: "+str(ave_recall/float(len(concepts)-split_num)))
print("average map is: "+str(ave_map/float(len(concepts)-split_num)))
