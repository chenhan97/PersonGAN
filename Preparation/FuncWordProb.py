import sys
sys.path.append("..")
from utils import Util
from nltk.tag import StanfordPOSTagger

model_filename = sys.path[0] + '/Preparation/models/english-bidirectional-distsim.tagger'
path_to_jar = sys.path[0] + '/Preparation/stanford-postagger.jar'
func_tag = ['WHD','WRB','WP','UH','MD','CC','DT','IN','EX','TO','WP$']

def FuncWordExtractor(files):
    tagger = StanfordPOSTagger(model_filename, path_to_jar)
    func_list = []
    for file in files:
        with open(file,'r',encoding='utf-8') as Reader:
            for line in Reader:
                sentence = tagger.tag(line.split())
                for WordTag in sentence:
                    if WordTag[1] in func_tag:
                        func_list.append()
                        if WordTag[0].lower() not in dict_POS[WordTag[1]].keys():
                            dict_POS[WordTag[1]][WordTag[0].lower()] = 1
                        else:
                            dict_POS[WordTag[1]][WordTag[0].lower()] += 1
    return dict_POS

def AuthorFuncWord(author_name):
    author_news = Util.listdir(sys.path[0]+"/Preparation/data/"+author_name) 
    print("function words have been recorded")
    return freq_to_prob(author_news,author_name)
    


