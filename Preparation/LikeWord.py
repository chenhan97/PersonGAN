import nltk
from os import remove
import sys
sys.path.append("..")
from utils import Util


def PerFreqDist(path):
    Text = open(path,encoding='utf-8')  
    text = Text.read()
    count_word = len([word for word in text.split()])
    fdist = nltk.FreqDist(word.lower() for word in text.split())
    Text.close()
    for word in fdist.keys():
        fdist[word] = fdist[word]/count_word
    return fdist

def AllFreqDist():
    FreqWord = {}
    authors_news = Util.listdir(sys.path[0]+"/data")
    authors_name = Util.listfile(sys.path[0]+"/data")
    for author in authors_news:
        author_name = authors_name[authors_news.index(author)]
        news = Util.listdir(author)
        for text in news:
            with open(text,'r',encoding='utf-8') as Reader:
                TempDist = PerFreqDist(text)
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
    remove(sys.path[0]+'/save/LikeWord')
    FreqWord = AllFreqDist()
    for word in FreqWord.keys():
        DifferFlag = False
        mean = sum(FreqWord[word].values())/len(FreqWord)
        for author in FreqWord[word].keys():
            if abs(FreqWord[word][author]-mean)/mean >= threshold:
                DifferFlag = True
                break
        with open(sys.path[0]+'/save/LikeWord','a') as Writer:
            if DifferFlag:
                Writer.write(word+"|||"+max(FreqWord[word],key=FreqWord[word].get)+"\n")
            else:
                Writer.write(word+"|||"+"AllUsed"+"\n")
    print("every word has been assigned to reporters' favorate vocabulary")
            

