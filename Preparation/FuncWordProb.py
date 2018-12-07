import sys
sys.path.append("..")
from utils import Util
from nltk.tag import StanfordPOSTagger

model_filename = sys.path[0] + '/models/english-bidirectional-distsim.tagger'
path_to_jar = sys.path[0] + '/stanford-postagger.jar'
func_tag = ['WHD','WRB','WP','UH','MD','CC','DT','IN','EX','TO','WP$']

def FuncWordExtractor(path):
    files = Util.listdir(path)
    tagger = StanfordPOSTagger(model_filename, path_to_jar)
    dict_POS = {}
    for tag in func_tag:
        dict_POS[tag] = {}
    for file in files:
        with open(file,'r',encoding='utf-8') as Reader:
            for line in Reader:
                sentence = tagger.tag(line.split())
                for WordTag in sentence:
                    if WordTag[1] in dict_POS.keys():
                        if WordTag[0].lower() not in dict_POS[WordTag[1]].keys():
                            dict_POS[WordTag[1]][WordTag[0].lower()] = 1
                        else:
                            dict_POS[WordTag[1]][WordTag[0].lower()] += 1
    return dict_POS

def freq_to_prob(path,author_name):
    freq_dict_POS = FuncWordExtractor(path)
    for POS in freq_dict_POS.keys():
        count = 0
        for word in freq_dict_POS[POS].keys():
            count = count + freq_dict_POS[POS][word]   
        for word in freq_dict_POS[POS].keys():
            freq_dict_POS[POS][word] = freq_dict_POS[POS][word]/count
    with open(sys.path[0]+'/save/FuncWordProb_'+author_name,'w') as Writer:
        for POS in freq_dict_POS.keys():
            Writer.write(POS+":  "+str(freq_dict_POS[POS])+'\n')
    return freq_dict_POS

def AuthorFuncWord(author_name):
    author_news = Util.listdir(sys.path[0]+"/data/"+author_name)
    print("function words have been recorded")
    return freq_to_prob(author_news,author_name)
    


