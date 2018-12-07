import nltk
import sys
sys.path.append("..")
from utils import Util

def TextNERExtractor(filepath):
    NER_word_list = []
    Text = open(filepath,encoding='utf-8')  
    text = Text.read()     
    tagged = nltk.pos_tag(text)   
    entities = nltk.chunk.ne_chunk(tagged)   
    for word in entities:
        if len(word)==1:
            NER_word_list.append(str(word[0][0]))
    Text.close()
    return list(set(NER_word_list))

def CompareNER(path, key_info_list, cover=1, min_corpu=50):
    files = Util.listdir(path)
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
        
def MergeQualFile(key_info_list):
    authors_news = Util.listdir(sys.path[0]+"/data")
    authors_name = Util.listfile(sys.path[0]+"/data")
    for author in authors_news:
        author_name = authors_name[authors_news.index(author)]
        QualifyFileList = CompareNER(author,key_info_list)
        for file in QualifyFileList:
            with open(file,'r',encoding='utf-8') as Reader, open(sys.path[0]+"/save/data"+author_name,'a',encoding='utf-8') as Writer:
                for line in Reader:
                    Writer.write(line)
    print("qulified files are all found")
