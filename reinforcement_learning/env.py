#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec
from gensim.models import Doc2Vec
from load import *
from find_stocks import *
import pickle
import jieba
from scipy import spatial

class Env:
	def __init__(self, use_doc=False):
		self.use_doc=use_doc
		#load word2vec models
		if use_doc:
			self.finance=Doc2Vec.load("../finance/vectors")
			self.wiki=Doc2Vec.load("../finance/vectors")
			self.bing=Doc2Vec.load("../bing/vectors")		
		else:
			self.finance=Word2Vec.load("../finance/vectors")
			self.wiki=Word2Vec.load("../finance/vectors")
			self.bing=Word2Vec.load("../bing/vectors")

		self.ranking=self.finance

		#load dictionary
		self.dictionary = pickle.load(open("dictionary", 'rb'))

		#all stocks
		self.stock={}
		load_stock(self.stock)

		#load concepts
		self.concepts={}
		self.concepts_set=[]
		
		load_concept(self.concepts, self.concepts_set, 'jinrongjie')

		#load description
		self.description={}
		load_description(self.description)

		#ratio of expansion word
		self.ratio=0.2

		#P@5 for valuation
		self.amount=5

	def reset(self, query):
		self.concept=query
		query=query.lower()

		seg_list = jieba.cut(query, cut_all=False)
		seg_list=list(seg_list)
		seg_list=[element for element in seg_list if element != query ]
		query_list=[query]+seg_list
		#update the candidate words for the query
		self.get_new_candidates(query_list)

		self.state=[query_list, self.get_next(self.top_finance, self.finance, query_list), \
		  self.get_next(self.top_bing, self.bing, query_list), \
		  self.get_next(self.top_wiki, self.wiki, query_list)]
		print("INITIAL STATE:")
		print(self.state[0])
		#print(self.state)
		return self.state

	def get_next(self, l, model, query_list):
		#get first candidate
		candidate=l.pop(0)[0]
		while isdigit(candidate) or candidate in query_list:
			try:
				candidate=l.pop(0)[0]
			except:
				#make the word unknown
				return ['厷']

		#extend it with its contexts
		if self.use_doc:
			#candidata's vector
			context=[get_word_vec(model, candidate)]
			
			score_list=[]			
			query_vector=get_word_vec(model, candidate+query_list[0])
			for doc_vec in model.docvecs:
				cos_similarity = 1 - spatial.distance.cosine(query_vector, doc_vec)
				score_list.append((cos_similarity, doc_vec))
			score_list.sort()
			top_n=score_list[-5:]
			for score, vector in top_n:
				context.append(vector)
			print(context)
		else:
			context=[candidate]
			for word in model.wv.most_similar(positive=[candidate], topn=5):
				if not isdigit(word[0]):
					context+=[word[0]]
		#extend it with its related stocks 
		#candidate_vec=self.get_vector_representation([candidate], self.ranking)
		#top=top_stock(candidate_vec, self.stock, self.ranking)[:5]
		#context+=top
		return context

	def step(self, action):
		reward=0
		stop=False
		if action<=2:
			prev_acc=self.map(self.state[0])	
			#prev_acc=self.top_rank_accuracy(self.state[0])		
			#prev_acc=self.rank_accuracy(self.state[0])		
			#prev_acc=self.accuracy(self.state[0])
			#prev_acc=self.micro_accuracy(self.state[0])
			#prev_acc=self.top_micro_accuracy(self.state[0])
			newquery=self.state[0]+ [self.state[action+1][0]]
			self.get_new_candidates(newquery)
			self.state=[newquery, self.get_next(self.top_finance, self.finance, newquery), self.get_next(self.top_bing, self.bing, newquery), self.get_next(self.top_wiki, self.wiki, newquery)]
			
			new_acc=self.map(newquery)
			#new_acc=self.top_rank_accuracy(newquery)
			#new_acc=self.rank_accuracy(newquery)
			#new_acc=self.accuracy(newquery)
			#new_acc=self.micro_accuracy(newquery)
			#new_acc=self.top_micro_accuracy(self.state[0])
			print("PREVIOUS ACCURACIES: "+str(prev_acc)+" NEW ACCURACIES: "+str(new_acc))
			
			reward=50*(new_acc-prev_acc)
			#reward=4*(prev_acc-new_acc)/float(prev_acc)
			#reward=2*(prev_acc-new_acc)/float(prev_acc)
			#reward=new_acc-prev_acc-0.005
			#reward=200*(new_acc-prev_acc)
			#reward=200*(new_acc-prev_acc)
		if action ==3:
			reward=0.0001
			self.state=[self.state[0], self.get_next(self.top_finance, self.finance, self.state[0]), self.get_next(self.top_bing, self.bing, self.state[0]), self.get_next(self.top_wiki, self.wiki, self.state[0])]
		if action == 4:
			stop=True
		print("NEW STATE:")
		print(self.state[0])
		#print(self.state)
		print("REWARD:")
		print(reward)
		return (self.state, reward, stop)

	def get_new_candidates(self, query_list):
		finance_vec=get_word_vec(self.finance, query_list[0])
		bing_vec=get_word_vec(self.bing, query_list[0])
		wiki_vec=get_word_vec(self.wiki, query_list[0])
		for query in query_list[1:]:
			finance_vec+=self.ratio*get_word_vec(self.finance,query)
			bing_vec+=self.ratio*get_word_vec(self.bing,query)
			wiki_vec+=self.ratio*get_word_vec(self.wiki,query)

		self.top_finance=self.finance.wv.similar_by_vector(finance_vec, topn=3000)
		self.top_bing=self.bing.wv.similar_by_vector(bing_vec, topn=3000)
		self.top_wiki=self.wiki.wv.similar_by_vector(wiki_vec, topn=3000)

		#print("NEW CANDIDATES:")
		#print(self.top_finance[:20])
		#print(self.top_bing[:20])
		#print(self.top_wiki[:20])

	def get_vector_representation(self, query_list, model):
		concept_vec=get_word_vec(model, query_list[0])
		for query in query_list[1:]:
			concept_vec+=self.ratio*get_word_vec(model,query)
		return concept_vec


	def accuracy(self, query_list, amount):
		concept_vec=self.get_vector_representation(query_list, self.ranking)
		#find stocks
		top=top_stock(concept_vec, self.stock, self.ranking)
		#print top 5 stocks
		print(top[:5])
		#compute precision
		precision=compute_precision(top, amount, self.concepts[self.concept])
		return precision

	def recall(self, query_list, num):
		concept_vec=self.get_vector_representation(query_list, self.ranking)
		#find stocks
		top=top_stock(concept_vec, self.stock, self.ranking)
		#compute recall
		r=compute_recall(top, num, self.concepts[self.concept])
		return r

	#MAP
	def map(self, query_list):
		concept_vec=self.get_vector_representation(query_list, self.ranking)
		#find stocks
		top=top_stock(concept_vec, self.stock, self.ranking)
		#compute precision
		m=compute_map(top, self.concepts[self.concept])
		return m

	#accuracy by considering stock similarities
	def micro_accuracy(self, query_list):
		concept_vec=self.get_vector_representation(query_list, self.ranking)
		#find stocks
		top=stock_simialrity(concept_vec, self.stock, self.ranking)
		#compute precision
		precision=compute_micro_precision(top, self.concepts[self.concept])
		return precision

	#accuracy by considering top stock similarities
	def top_micro_accuracy(self, query_list):
		concept_vec=self.get_vector_representation(query_list, self.ranking)
		#find stocks
		top=stock_simialrity(concept_vec, self.stock, self.ranking)
		#compute precision
		precision=compute_micro_amount_precision(top, self.concepts[self.concept], self.amount)
		return precision

	#accuracy by considering stock ranks
	def rank_accuracy(self, query_list):
		concept_vec=self.get_vector_representation(query_list, self.ranking)
		#find stocks
		top=top_stock(concept_vec, self.stock, self.ranking)
		#compute precision
		precision=compute_rank_precision(top, self.concepts[self.concept])
		return precision

	#accuracy by considering stock top ranks 
	def top_rank_accuracy(self, query_list):
		concept_vec=self.get_vector_representation(query_list, self.ranking)
		#find stocks
		top=top_stock(concept_vec, self.stock, self.ranking)
		#compute precision
		precision=compute_rank_amount_precision(top, self.concepts[self.concept], self.amount)
		return precision
#e=Env()
#state=e.reset("中字头")
#print("ACTION:")
#print(2)
#newstate, reward, stop=e.step(2)
#newstate, reward, stop=e.step(2)
