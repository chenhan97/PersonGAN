# PersonGAN

*******************************************************************************************************************************
PersonGAN is a text generation model based on LeakGAN. This model can generate sentences with personalized writing style. 
At the same time, this model also possesses the ability to generate sentences related to user-defined topic.

If you got any questions or suggestions when using this code, you are welcomed to contact chris.yuan.ece@gmail.com or yuanchenhan@std.uestc.edu.cn
*******************************************************************************************************************************
## Requirements
  * Ubuntu(16.04)
  * Tensorflow(>=1.11.0)
  * gensim
  * colorama
  * NLTK(>=3.4)
  * Java Runtime Environment(JRE)
  * CUDA(>=7.5)(For GPU)
  * Python(>=3.5)

********************************************************************************************************************************
## Start

You can start training by typing `python3 main.py` in the command line to run the main.py file.

  When starting training model, you are asked to enter named entities to determine the topic of generated text. 

  (Note: One space between each named entity and all named entities are lowercase)

  Then, you are asked to enter an author's name to let the model extract personalized information from the author's text.

  (For example, you can use the text stored in `/PersonGAN/Preparation/data/Simon_Denyer` as a training corpus, so you need to enter `Simon_Denyer` at this stage to specify the search path for the program.)

A sample of correct training process：
```script
$ python3 main.py
enter NER information:(make sure there is space between each NER)
linkinpark chester bennington transformers
enter reporter's name
Simon_Denyer
Source is limited, please show more relevant news
qulified files are all found
......
```
*********************************************************************************************************************************
## Result

After every 5 epochs, PersonGAN will generate test data named `final_time` and will save that in `/PersonGAN/save`.
