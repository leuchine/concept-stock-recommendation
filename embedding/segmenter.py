#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import os
from bs4 import *
from hanziconv import HanziConv
import re
import jieba
import sys

location="../"+sys.argv[1]

for concept in os.listdir(location+"/concept"):
	if concept != '.DS_Store':
		try:
			os.mkdir(location+"/output/"+concept)
		except:
			pass		
		for file in os.listdir(location+"/concept/"+concept):
			if file != '.DS_Store':
				with open(location+"/concept/"+concept+"/"+file) as f:
					content=f.read()
					seg_list = jieba.cut(content, cut_all=False)
					seg_content=" ".join(seg_list)
					with open(location+"/output/"+concept+"/"+file, 'w') as f1:
						f1.write(seg_content)
