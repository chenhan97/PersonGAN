import nltk
from os import remove,path
import sys
sys.path.append("..")
from utils import Util

def PerFreqDist(path):
    news = Util.listdir(path)
    count_word = 0
    FreqWord = {}    
    for new in news:
        Text = open(new,encoding='utf-8')  
        text = Text.read()
        count_word = len([word for word in text.split()]) + count_word
        Text.close()
    for new in news:
        Text = open(new,encoding='utf-8')  
        text = Text.read()
        Text.close()        
        fdist = nltk.FreqDist(word.lower() for word in text.split())
        for word in fdist:
            if word in FreqWord.keys():
                FreqWord[word] = FreqWord[word] + 1
            else:
                FreqWord[word] = 1
    for word in FreqWord.keys():
        FreqWord[word] = FreqWord[word]/count_word
    return FreqWord

def AllFreqDist():
    FreqWord = {}
    authors_news = Util.listdir(sys.path[0]+"/Preparation/data")
    authors_name = Util.listfile(sys.path[0]+"/Preparation/data")
    for author in authors_news:
        author_name = authors_name[authors_news.index(author)]
        TempDist = PerFreqDist(author)
        for word in TempDist:
            if word not in FreqWord.keys():
                FreqWord[word] = {}
                FreqWord[word][author_name] = TempDist[word]
            else:
                if author_name not in FreqWord[word].keys():
                    FreqWord[word][author_name] = TempDist[word]
                else:
                    FreqWord[word][author_name] += TempDist[word]
    return FreqWord

def GenerateVocab(threshold=0.3):
    if path.exists(sys.path[0]+'/Preparation/save/data/LikeWord'):   
        remove(sys.path[0]+'/Preparation/save/LikeWord')
    FreqWord = AllFreqDist()
    for word in FreqWord.keys():
        DifferFlag = False
        mean = sum(FreqWord[word].values())/len(FreqWord[word].values())
        for author in FreqWord[word].keys():
            if abs(FreqWord[word][author]-mean)/mean >= threshold:
                DifferFlag = True
                break
        with open(sys.path[0]+'/Preparation/save/LikeWord','a') as Writer:
            if DifferFlag:
                Writer.write(word+" ||| "+max(FreqWord[word],key=FreqWord[word].get)+"\n")
            else:
                Writer.write(word+" ||| "+"AllUsed"+"\n")
    print("every word has been assigned to reporters' favorate vocabulary")
            

