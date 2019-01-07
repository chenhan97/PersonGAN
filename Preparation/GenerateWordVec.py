import gensim
from os import remove,path
import sys
sys.path.append("..")

class MySentences(object): 
    def __init__(self, fname): 
        self.fname = fname 
    def __iter__(self): 
        for line in open(self.fname):
            yield line.split()

def GetWordVec(author_name):
    sentences = MySentences(sys.path[0]+"/Preparation/save/data/"+author_name)
    model = gensim.models.word2vec.Word2Vec(sentences,size = 100,hs = 1,min_count=5,window =5)
    model.save(sys.path[0]+"/Preparation/save/wordvec_model")
    print("generating the word's vector has done")