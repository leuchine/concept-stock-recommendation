#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import os
from bs4 import *
from bs4 import Comment
from hanziconv import HanziConv
import re
import sys

location="../"+sys.argv[1]
for concept in os.listdir(location+"/concept"):
	if concept != '.DS_Store':
		for file in os.listdir(location+"/concept/"+concept):
			if file =='.DS_Store':
				continue
			
			with open(location+"/concept/"+concept+"/"+file) as f:
				content=f.read()
				soup = BeautifulSoup(content)
				for script in soup.find_all('script', src=False):
					script.decompose()
				for style in soup.find_all('style'):
					style.extract()
				for comment in soup.findAll(text=lambda text:isinstance(text, Comment)):
					comment.extract()
				try:
					text = ''.join(soup.body.findAll(text=True))
				except:
					print(location+"/concept/"+concept+"/"+file)
					continue
				text = HanziConv.toSimplified(text)
				text = text.replace('\n',' ')

				with open(location+"/concept/"+concept+"/"+file, 'w') as f2:
					f2.write(text)
