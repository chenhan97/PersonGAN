from nltk.tag import StanfordNERTagger
import sys
sys.path.append("..")
from utils import Util
import os

def TextNERExtractor(filepath):
    NERList = ['LOCATION', 'PERSON', 'ORGANIZATION', 'MONEY', 'PERCENT', 'DATE', 'TIME']
    model_filename = sys.path[0] + '/Preparation/models/english.muc.7class.distsim.crf.ser.gz'
    path_to_jar = sys.path[0] + '/Preparation/stanford-ner.jar'
    NER_word_list = []
    Text = open(filepath,encoding='utf-8')  
    text = Text.read()  
    tagged = StanfordNERTagger(model_filename,path_to_jar)   
    entities = tagged.tag(text.split())
    for word in entities:
        if word[1] in NERList:
            NER_word_list.append(word[0])
    Text.close()
    return list(set(NER_word_list))

def CompareNER(path, key_info_list, cover=1, min_corpu=500): #change for test
    files = Util.NERlist(path)
    train_file = []
    for file in files:
        NER_word_list = TextNERExtractor(file)
        if len([word for word in NER_word_list if word in key_info_list]) >=cover:
            train_file.append(file)
    if len(train_file) < min_corpu:
        print("Source is limited, please show more relevant news")
        from random import sample
        extend_train_file = sample([file for file in files if file not in train_file], min_corpu - len(train_file))
        train_file = train_file + extend_train_file
    return train_file
        
def MergeQualFile(key_info_list,author_name,Cover,Min_corpu):
    QualifyFileList = CompareNER(sys.path[0]+"/Preparation/data",key_info_list,cover=Cover,min_corpu = Min_corpu)
    for file in QualifyFileList:
        with open(file,'r',encoding='utf-8') as Reader, open(sys.path[0]+"/Preparation/save/data/"+author_name,'a',encoding='utf-8') as Writer:
            for line in Reader:
                Writer.write(line)
    print("qulified files are all found")
    
