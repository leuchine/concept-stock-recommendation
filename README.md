concept-stock-recommendation
======
A framework for mining relevant stocks given a topic of concern on emerging capital markets. 

Prerequisite
======
1. Tensorflow 
2. Google News Embeddings (https://code.google.com/archive/p/word2vec/) (rename it to 'vectors.gz')
3. Gensim
4. Jieba
5. BeautifulSoup 4
6. Hanziconv
7. Scikit-Learn

Data Preparation
======
1. Download financial documents as datasets. We use financial news, annual reports, Wikipedia pages and Bing search reports in the paper. However, any document set that could draw associations between concepts and stocks can be used.

2. Organize the documents into dataset_name/concept/concept_name/documents (e.g. wikipedia/concept/3D打印/documents)

2. Preprocessing: run `python embedding/remove_tags.py dataset_name`, `python embedding/segmenter.py dataset_name` and `python embedding/preprocessing.py dataset_name` in sequence to preprocess the documents. 

Train Embeddings and Reinforcement Learning
======

1. To train embeddings, either run `python embedding/embedding.py dataset_name` or `python embedding/doc_embedding.py dataset_name`. The first one uses Gensim's Word2Vec and the second one uses Gensim's Doc2Vec.

2. Run `python reinforcement_learning/learning.py` for training the reinforcement learning.

3. Run `python embedding/accuracy.py` if you would like to directly use trained embeddings for ranking stocks without using reinforcement learning.

=====
Due to copyright issues, the raw datasets and trained embeddings are not open. Email me (qiliu@u.nus.edu) if you would like to have some samples.

Sample Output
======
Query: 铜
Results: 云南铜业, 江西铜业, 锡业股份, 宏达股份, 中金岭南, 恒邦股份, 西部矿业, 中金黄金, 驰宏锌锗

Query: 房地产
Results: 招商地产, 滨江集团, 世荣兆业, 绿景地产, 万科, 大龙地产, 苏宁环球, 上实发展, 阳光发展

Query: 物联网
Results: 远望谷, 新大陆, 星网锐捷, 华胜天成, 卫士通, 烽火通信, 英唐智控, 银河电子, 超声电子


