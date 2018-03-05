#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import os
from bs4 import *
from hanziconv import HanziConv
import re
import jieba
import sys

def replace_wrong_segment(sentence, name):
	for gap in range(len(name)-1):
		seg_name= name[:gap+1]+" "+name[gap+1:]
		sentence=sentence.replace(seg_name, name)	
	return sentence

with open("news.txt") as f:
	with open("seg_news.txt",'w') as f2:
		for line in f:
			try:
				seg_list = jieba.cut(line.split('||')[2], cut_all=False)
				seg_content=" ".join(seg_list).lower()
				seg_content=replace_wrong_segment(seg_content, line.split('||')[0])
				f2.write(line.split('||')[0]+'||'+seg_content)
			except:
				pass
