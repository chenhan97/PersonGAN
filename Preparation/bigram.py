import os
import itertools
from nltk.tag import StanfordPOSTagger
import sys
sys.path.append("..")
from utils import Util

model_filename = sys.path[0]+'/Preparation/models/english-bidirectional-distsim.tagger'
path_to_jar = sys.path[0]+'/Preparation/stanford-postagger.jar'
func_tag = ['WHD','WRB','WP','UH','MD','CC','DT','IN','EX','TO','WP$']

def Get2gram(author_name):
    word_dict = {}
    files = Util.listdir(sys.path[0]+'/Preparation/data/'+author_name)
    tagger = StanfordPOSTagger(model_filename, path_to_jar)
    a = "CC CD DT EX FW IN JJ JJR JJS LS MD NN NNS NNP NNPS PDT POS PRP PRP$ RB RBR RBS RP SYM TO UH VB VBD VBG VBN VBP VBZ WDT WP WP$ WRB , $ :"
    Temp = [x for x in range(len(a.split()))]
    tag_dict = {}
    tag = a.split() 
    func_list = []
    for i in itertools.product(Temp,repeat = 2):
        tag_dict[tag[i[0]]+" "+tag[i[1]]] = 0    
    for file in files:
        with open(file,'r',encoding='utf-8') as Reader:
            for index,line in enumerate(Reader):
                sent_real = line.split()
                sent_tag = tagger.tag(sent_real)
                for WordTag in sent_tag:
                    if WordTag[1] in func_tag:
                        func_list.append(WordTag[0])                
                for i in range(len(sent_tag)-1):
                    tag_dict[sent_tag[i][1]+' '+sent_tag[i+1][1]] = tag_dict[sent_tag[i][1]+' '+sent_tag[i+1][1]] + 1  
        func_list = list(set(func_list))
    for tag in tag_dict.keys():
        if tag_dict[tag]==0:
            tag_dict[tag] = 0.5
    count = sum(tag_dict.values())
    for tag in tag_dict.keys():
        tag_dict[tag] = tag_dict[tag]/count
    
    with open(sys.path[0]+'/Preparation/save/2gram_tag_'+author_name,'w',encoding='utf-8') as Writer:
        for tag in tag_dict.keys():
            Writer.write(str(tag)+':'+str( tag_dict[tag])+'\n')  
    return tag_dict,func_list