#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np


def load_stock(stock):
	with open("../wiki/all_stocks.txt") as f:
		for i in f:
			elements=i.split(",")
			if len(elements) == 7 or len(elements) == 10:
				stock[elements[1]]=elements[0]
			elif len(elements) == 6 or len(elements) == 5 or len(elements) == 4 or len(elements) == 8:
				stock[elements[2]]=elements[1]
			else:
				continue
				
def load_concept(concepts, concept_set, data_source):
	if data_source=='jinrongjie':
		with open("../jinrongjie/related_stocks.txt") as f:
			for line in f:
				concept=line.split("||")[0]
				stock=line.split("||")[2]
				
				concepts[concept]=concepts.get(concept, [])+[stock]

				if concept not in concept_set:
					concept_set.append(concept)
	else:
		with open("../tonghuashun/related_stock.txt") as f1:
			with open("../tonghuashun/label.txt") as f2:
				for f1_line in f1:
					concept=f2.readline().strip()
					stock=f1_line.strip().split("||")[1]

					if stock not in concepts.get(concept, []):
						concepts[concept]=concepts.get(concept, [])+[stock]
				
					if concept not in concept_set:
						concept_set.append(concept)


def load_description(descriptoon):
	with open("../jinrongjie/seg_news.txt") as f:
		for line in f:
			line=line.strip()
			concept=line.split("||")[0]
			content=line.split("||")[1].split()		
			descriptoon[concept]=content

def get_word_vec(model, concept):
	#whole string
	concept_vec=np.array([0.0]*300)
	try:
		concept_vec=model.wv[concept]
	except:
		pass

	#bigram
	if np.sum(concept_vec==np.array([0.0]*300))==300:
		for i in range(len(concept)-1):
			try:
				if concept_vec is None:
					concept_vec=model.wv[concept[i:i+2]]
				else:
					concept_vec+=model.wv[concept[i:i+2]]
			except:
				pass
	#letter
	if np.sum(concept_vec==np.array([0.0]*300))==300:
		for i in range(len(concept)):
			try:
				if concept_vec is None:
					concept_vec=model.wv[concept[i]]
				else:
					concept_vec+=model.wv[concept[i]]
			except:
				pass
	return np.array(concept_vec)

#concept={}
#concept_set=[]
#load_concept(concept, concept_set, 'tonghuashun')
#print(concept)