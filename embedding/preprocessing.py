#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from load import *
import sys

def replace_wrong_segment(sentence, name):
	for gap in range(len(name)-1):
		seg_name= name[:gap+1]+" "+name[gap+1:]
		sentence=sentence.replace(seg_name, name)	
	return sentence

#load stocks
stock={}
load_stock(stock)

#load concepts
concepts={}
load_concept(concepts, [])

counter=0

location="../"+sys.argv[1]+"/output/"
for i in os.listdir(location):
	
	if i=='.DS_Store':
		continue
	print(i)
	for file in os.listdir(location+"/"+i):
		with open(location+"/"+i+'/'+file) as f:
			content=f.read()
			sentences=[content.replace("\n", " ")]
			with open("../"+sys.argv[1]+"/sentence"+"/"+str(counter), 'w') as fw:
				
				for sentence in sentences:				
					for name, code in stock.items():
						sentence=sentence.replace(code, name)
						sentence=replace_wrong_segment(sentence, name)

					for concept in concepts:
						sentence=replace_wrong_segment(sentence, concept)
					fw.write(sentence.strip()+"\n")	
				counter+=1
				if counter %100==0:
					print(counter)