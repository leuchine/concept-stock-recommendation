#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from bs4 import BeautifulSoup
from selenium import webdriver

chromedriver = "../chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome(chromedriver)

#all concepts involved
urls=['https://zh.wikipedia.org/wiki/%E6%B7%B1%E5%9C%B3%E8%AF%81%E5%88%B8%E4%BA%A4%E6%98%93%E6%89%80%E5%88%9B%E4%B8%9A%E6%9D%BF%E4%B8%8A%E5%B8%82%E5%85%AC%E5%8F%B8%E5%88%97%E8%A1%A8','https://zh.wikipedia.org/wiki/%E6%B7%B1%E5%9C%B3%E8%AF%81%E5%88%B8%E4%BA%A4%E6%98%93%E6%89%80%E4%B8%8A%E5%B8%82%E5%85%AC%E5%8F%B8%E5%88%97%E8%A1%A8','https://zh.wikipedia.org/wiki/%E4%B8%8A%E6%B5%B7%E8%AF%81%E5%88%B8%E4%BA%A4%E6%98%93%E6%89%80%E4%B8%8A%E5%B8%82%E5%85%AC%E5%8F%B8%E5%88%97%E8%A1%A8','https://zh.wikipedia.org/wiki/%E6%B7%B1%E5%9C%B3%E8%AF%81%E5%88%B8%E4%BA%A4%E6%98%93%E6%89%80%E4%B8%AD%E5%B0%8F%E6%9D%BF%E4%B8%8A%E5%B8%82%E5%85%AC%E5%8F%B8%E5%88%97%E8%A1%A8']


with open("all_stocks.txt", 'w') as f:
	for url in urls:	
		print()
		print("start processing: "+url)
		f.write(url+"\n")
		driver.get(url)
		html = driver.page_source.encode('utf-8')
		soup= BeautifulSoup(html, 'html.parser')

		for table in soup.find_all('table', {"class": "wikitable"}):
			for row in table.find_all('tr'):
				line=[]
				for data in row.find_all('td'):
					line+=[data.get_text()]

				f.write(','.join(line)+"\n")
driver.quit()
