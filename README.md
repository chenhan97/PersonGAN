# PersonGAN

*******************************************************************************************************************************
Author: Chenhan Yuan
Email: chris.yuan.ece@gmail.com; yuanchenhan@std.uestc.edu.cn 
*******************************************************************************************************************************
PersonGAN is a text generation model based on LeakGAN. This model can generate sentences with personalized writing style. 
At the same time, this model also possesses the ability to generate sentences related to user-defined topic.
This code is the source code of PersonGAN model proposed by Mr. Chenhan Yuan and Prof. Yi-chin Huang. 
*******************************************************************************************************************************
# Requirements
Ubuntu(16.04)

Tensorflow(>=1.11.0)

gensim

NLTK(>=3.4)

Java Runtime Environment(JRE)

CUDA(>=7.5)(For GPU)

Python(>=3.5)

********************************************************************************************************************************
# Start

You can start training by typing `python3 main.py` in the command line to run the main.py file.
You need to enter named entities to determine the topic of generated text. (Note: One space between each named entity and all named entities are lowercase)

# Result

After every 5 epochs, PersonGAN will generate test data named `final_time` and save that in `/PersonGAN/save`.
