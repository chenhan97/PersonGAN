import getopt
import sys
from colorama import Fore
from models.persongan import Persongan
from Preparation import SimilarityNER
from Preparation import FuncWordProb
from Preparation import LikeWord
from Preparation import GenerateTagVec
from Preparation import GenerateWordVec
from Preparation import WordTagProb
from Preparation import bigram

def set_training(gan):
    gan_func = gan.train_real
    return gan_func
 
if __name__ == '__main__':
    print("enter NER information:(make sure there is space between each NER)")
    key_info_list = str(input()).split()
    print("enter reporter's name")
    AuthorName = str(input())
    SimilarityNER.MergeQualFile(key_info_list,AuthorName, 1, 500)
    GenerateTagVec.GetTagVec(AuthorName,32)
    LikeWord.GenerateVocab()
    GenerateWordVec.GetWordVec(AuthorName)
    WordTag = WordTagProb.GetTagVec(AuthorName)
    tag_bi,FuncWordList = bigram.Get2gram(AuthorName)
    print("preparation process finished")
    gan = Persongan.Persongan()  
    gan_func = set_training(gan)
    gan_func(sys.path[0]+'/Preparation/save/data/'+AuthorName,author_name=AuthorName,StrucWord=FuncWordList,WordTagPro=WordTag, TagBi=tag_bi)  
