import gensim
from nltk.tag import StanfordPOSTagger
from os import remove,path
import sys
sys.path.append("..")

model_filename = sys.path[0] + '/Preparation/models/english-bidirectional-distsim.tagger'
path_to_jar = sys.path[0] + '/Preparation/stanford-postagger.jar'

class MySentences(object): 
    def __init__(self, fname): 
        self.fname = fname 
    def __iter__(self): 
        for line in open(self.fname):
            yield line.split()

def Text_to_tag(Readfile,file):
    if path.exists(sys.path[0]+'/Preparation/save/data/'+file):
        remove(sys.path[0]+'/Preparation/save/data/'+file)
    tagger = StanfordPOSTagger(model_filename, path_to_jar)
    for line in open(Readfile):
        TagList = []
        sentence = tagger.tag(line.split())
        for WordTag in sentence:
            TagList.append(WordTag[1])
        with open(file,'a',encoding='utf-8') as Writer:
            Writer.write(" ".join(TagList)+'\n')

def GetTagVec(author_name,size):
    Text_to_tag(sys.path[0]+"/Preparation/save/data/"+author_name,sys.path[0]+"/Preparation/save/data/"+author_name+"_tag")
    sentences = MySentences(sys.path[0]+"/Preparation/save/data/"+author_name+"_tag")
    model = gensim.models.word2vec.Word2Vec(sentences,size = size,hs = 1,min_count=1,window =5)
    model.save(sys.path[0]+"/Preparation/save/tagvec_model")
    print("generating the tag's vector has done")


