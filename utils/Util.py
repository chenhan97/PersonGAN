import os
import sys
sys.path.append("..")

def listdir(path):
    files = []
    for file in os.listdir(path):
        files.append(os.path.join(path,file))
    return files

def listfile(path):
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files

def NERlist(file_dir):
    file = []
    for root, dirs, files in os.walk(file_dir):
        for dire in dirs:
            file = file + listdir(file_dir+"/"+dire)
    return file

            