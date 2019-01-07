from nltk.tag import StanfordPOSTagger
from os import remove,path
import sys
sys.path.append("..")

model_filename = sys.path[0] + '/Preparation/models/english-bidirectional-distsim.tagger'
path_to_jar = sys.path[0] + '/Preparation/stanford-postagger.jar'

def TagProb(Readfile,file):
    if path.exists(sys.path[0]+'/Preparation/save/data/'+file):
        remove(sys.path[0]+'/Preparation/save/data/'+file)
    tagger = StanfordPOSTagger(model_filename, path_to_jar)
    WordDict = {}
    for line in open(Readfile):
        sentence = tagger.tag(line.split())
        for WordTag in sentence:
            if WordTag[0] not in WordDict.keys():
                WordDict[WordTag[0]] = {}
                WordDict[WordTag[0]][WordTag[1]] = 1
            else:
                if WordTag[1] not in WordDict[WordTag[0]].keys():
                    WordDict[WordTag[0]][WordTag[1]] = 1
                else:
                    WordDict[WordTag[0]][WordTag[1]] = 1 + WordDict[WordTag[0]][WordTag[1]]
    for word in WordDict.keys():
        sum_freq = 0
        for tag in WordDict[word].keys():
            sum_freq = WordDict[word][tag] + sum_freq
        for tag in WordDict[word].keys():
            WordDict[word][tag] = WordDict[word][tag]/sum_freq
    with open(file,'a',encoding='utf-8') as Writer:
        for word in WordDict.keys():
            Writer.write(str(word)+':'+str( WordDict[word])+'\n')
    return WordDict

def GetTagVec(author_name):
    TagProbDict = TagProb(sys.path[0]+"/Preparation/save/data/"+author_name,sys.path[0]+"/Preparation/save/tagprob")
    print(TagProbDict)
    return TagProbDict