# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 23:21:51 2021

@author: Hanning Su
"""

from gensim import utils
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords, stem_text
from nltk.stem import WordNetLemmatizer as lemma #stemming or lemmatization or stopwords removal
import os
import nltk
import csv
import re
import string
import pattern
import logging

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#%%
os.chdir(os.path.dirname(__file__)) #set working directory to that of this script file

#%%
#We determine a text clean up procedure in this step
filename = "patent_abstract_title.csv"

lines = []

with open(filename, "r", encoding='cp932', errors='ignore') as patent_abstract_title:
    reader = csv.DictReader(patent_abstract_title)
    counter = 0
    for row in reader:
        if row["abstract"] == 'NULL':
            lines.append([row["title"]]) #Some patents has NULL abstraction, we will only use title
        else:
            lines.append([row["title"] + ' ' + row["abstract"]])
        counter += 1
        if counter > 10:
            break
 
#%%
#print(lines) #inspect first ten title + abstraction

test = re.sub('[^a-zA-Z0-9]', ' ', lines[6][0]) #replace non-letters and numbers with blank space

test1 = test.lower() #lower case

test2 = utils.simple_preprocess(test1) #tokenization
 
print(test2)

#%%
class MySentences(object):
    def __init__(self,filename):
        self.filename = filename
 
    def __iter__(self):
        with open(self.filename, "r", encoding='cp932', errors='ignore') as patent_abstract_title:
            reader = csv.DictReader(patent_abstract_title)
            for row in reader:
                if row["abstract"] == 'NULL':
                    line = re.sub('[^a-zA-Z0-9]', ' ', row["title"])
                    yield utils.simple_preprocess(line) #Some patents has NULL abstraction, we will only use title
                else:
                    line = re.sub('[^a-zA-Z0-9]', ' ', row["title"] + ' ' + row["abstract"])
                    yield utils.simple_preprocess(line)
 
input_word2vec = MySentences("patent_abstract_title.csv") # a memory-friendly iterator

#%%
# build vocabulary and train model 
#CBOW used
model = Word2Vec(
    input_word2vec,
    size = 200, #Set word vectors to be 200 dimensional
    window = 5, #per Li et al set window size to be 5
    min_count = 5, #per Li et al ignore words whose occurance frequency are below 5
    workers = 10, #how many threads to use behind the scenes
    iter = 10) # 10 epochs over the corpus

model.train(input_word2vec,total_examples = 7131735, epochs = 10)

#%%
#Save the model for potential further training
model.save("word2vec.model")

# Store just the words + their trained embeddings.
word_vectors = model.wv
word_vectors.save("word2vec.wordvectors")

# Load back with memory-mapping = read-only, shared across processes.
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')

del model
#%%
# build vocabulary and train model 
#Skip-gram and negative sampleing used
model = Word2Vec(
    input_word2vec,
    size = 200, #Set word vectors to be 200 dimensional
    window = 5, #per Li et al set window size to be 5
    min_count = 5, #ignore words whose occurance frequency are below 5
    workers = 10, #how many threads to use behind the scenes
    sg = 1, #use skip-gram instead of CBOW
    hs = 0, 
    negative = 3, #Mikolov 2013:  "Our experiments indicate that values of k
    #in the range 5–20 are useful for small training datasets, 
    #while for large datasets the k can be as small as 2–5." 
    iter = 10) # 10 epochs over the corpus

model.train(input_word2vec,total_examples = 7131735, epochs = 10)

#%%
#Save the model for potential further training
model.save("word2vec_sg.model")

# Store just the words + their trained embeddings.
word_vectors = model.wv
word_vectors.save("word2vec_sg.wordvectors")

# Load back with memory-mapping = read-only, shared across processes.
wv = KeyedVectors.load("word2vec_sg.wordvectors", mmap='r')

del model

#%%
### Example from gensim documentation
from gensim.test.utils import datapath

class MyCorpus:
    
    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            yield utils.simple_preprocess(line)
            
#%%
sentences = MyCorpus()
model = Word2Vec(sentences=sentences)

#%%
corpus_path = datapath('lee_background.cor')
for line in open(corpus_path):
    print(utils.simple_preprocess(line))
    
## Example of a input to W2V model loks like:
#['team', 'of', 'australian', 'and', 'israeli', 'scientists', 'have', 'conducted', 'what', 
#'they', 'believe', 'is', 'successful', 'research', 'into', 'using', 'human', 'embryo', 'cells', 
#'to', 'repair', 'brain', 'damage', 'but', 'their', 'findings', 'have', 'been', 'released', 'just', 
#'days', 'after', 'us', 'president', 'george', 'bush', 'criticised', 'similar', 'research', 'by', 
#'team', 'of', 'americans', 'earlier', 'this', 'week', 'massachusetts', 'based', 'company', 'advance', 
#'cell', 'technologies', 'said', 'it', 'had', 'successfully', 'cloned', 'an', 'early', 'stage', 'human', 
#'embryo', 'the', 'announcement', 'sparked', 'recriminations', 'from', 'us', 'congressmen', 'with', 'president', 
#'bush', 'saying', 'he', 'was', 'per', 'cent', 'against', 'any', 'type', 'of', 'human', 'cloning', 'now', 
#'an', 'australian', 'israeli', 'team', 'has', 'used', 'excess', 'ivf', 'embryos', 'to', 'create', 'precursor', 
#'brain', 'cells', 'which', 'they', 'injected', 'into', 'the', 'brains', 'of', 'baby', 'mice', 'the', 'findings', 
#'show', 'the', 'brain', 'cells', 'grew', 'to', 'be', 'from', 'other', 'brain', 'tissue', 'while', 'the', 
#'research', 'could', 'prove', 'useful', 'in', 'treating', 'variety', 'of', 'conditions', 'including', 
#'parkinson', 'disease', 'it', 'is', 'likely', 'to', 'come', 'under', 'fire', 'from', 'human', 'rights', 
#'groups', 'as', 'it', 'involves', 'the', 'destruction', 'of', 'human', 'embryos']


corpus_path = datapath('lee_background.cor')
for line in open(corpus_path):
    print(line)
    
## Exmaple of an input without tokenization
#A team of Australian and Israeli scientists have conducted what they believe is successful research 
#into using human embryo cells to repair brain damage. But their findings have been released just days 
#after US president George W Bush criticised similar research by a team of Americans. 
#Earlier this week Massachusetts based company Advance Cell Technologies said it had successfully 
#cloned an early stage human embryo. The announcement sparked recriminations from US Congressmen
# with President Bush saying he was 100 per cent against any type of human cloning. 
#Now an Australian-Israeli team has used excess IVF embryos to create precursor brain cells 
#which they injected into the brains of baby mice. The findings show the brain cells 
#grew to be indistinguishable from other brain tissue. While the research could prove useful in 
#treating a variety of conditions including Parkinson's disease, it is likely to come under fire 
#from human rights groups as it involves the destruction of human embryos. 
