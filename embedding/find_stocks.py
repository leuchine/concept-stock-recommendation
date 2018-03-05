#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.metrics.pairwise import *
import numpy 
#get top stocks by word vector similarities
def top_stock(concept_vec, all_stock, model):
	rank_list=[]
	for name in all_stock:
		try:
			stock_vector=model.wv[name]
			sim=cosine_similarity(numpy.array(concept_vec).reshape([1,-1]), numpy.array(stock_vector).reshape([1,-1]))[0][0]
			rank_list+=[(sim, name)]
		except KeyError:
			continue
	rank_list.sort()
	rank_list.reverse()
	top=[]
	for i in rank_list:
		top+=[i[1]]
	return top

def compute_map(top, related_stock):
	ranks=[]
	for index, stock in enumerate(top):
		index+=1
		if stock in related_stock:
			ranks.append(index)
	count=1
	precision_sum=0
	for index in ranks:
		precision_sum+=count/float(index)
		count+=1
	return precision_sum/float(len(ranks))

#precision
def compute_precision(top, amount, related_stock):
	intersection=0
	for stock in top[:amount]:
		if stock in related_stock:
			intersection+=1
	return intersection/float(amount)

#recall 
def compute_recall(top, amount, related_stock):
	intersection=0
	for stock in top[:amount]:
		if stock in related_stock:
			intersection+=1
	return intersection/float(len(related_stock))

def isdigit(s):
	try:
		float(s)
	except:
		return False
	return True