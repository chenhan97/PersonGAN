from time import time
import sys
sys.path.append("..")
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from models.Gan import Gan
from models.persongan.PersonganDataLoader import DataLoader, DisDataloader
from models.persongan.PersonganDiscriminator import Discriminator
from models.persongan.PersonganGenerator import Generator
from models.persongan.PersonganReward import Reward
from utils.metrics.Bleu import Bleu
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.oracle.OracleLstm import OracleLstm
from utils.utils import *
from utils import Util

def pre_train_epoch_gen(sess, trainable_model, data_loader):   
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()
    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, _, _ = trainable_model.pretrain_step(sess, batch, .8)
        supervised_g_losses.append(g_loss)
    return np.mean(supervised_g_losses)

def LikeWordIndex(w_i_dict,author_name):
    wi_like = [] # [word_index,word_index2]
    with open(r'Preparation/save/LikeWord','r',encoding='utf-8') as Reader:
        for line in Reader:
            word_author = line.split()
            if word_author[2]==author_name and word_author[0] in w_i_dict.keys():
                wi_like.append(w_i_dict[word_author[0]])
    return wi_like

def NoLikeWordIndex(w_i_dict,author_name):
    wi_nolike = [] # [word_index,word_index2]
    with open(r'Preparation/save/LikeWord','r',encoding='utf-8') as Reader:
        for line in Reader:
            word_author = line.split()
            if word_author[2]!=author_name and word_author[0] in w_i_dict.keys() and word_author[2]!="AllUsed":
                wi_nolike.append(w_i_dict[word_author[0]])
    return wi_nolike

def GetStructIndex(w_i_dict,StrucWord):
    wi_struc = [] #[word_index1, word_index2]
    for word in StrucWord:
        if word in w_i_dict.keys():
            wi_struc.append(w_i_dict[word])
    return wi_struc

def GetWordTag(w_i_dict,WordTagProb,emb_dim):
    import numpy as np
    TagVec = gensim.models.Word2Vec.load(sys.path[0] + r'/Preparation/save/tagvec_model')
    t_i_dict = {}
    i_t_dict = {}
    a = "CC CD DT EX FW IN JJ JJR JJS LS MD NN NNS NNP NNPS PDT POS PRP PRP$ RB RBR RBS RP SYM TO UH VB VBD VBG VBN VBP VBZ WDT WP WP$ WRB , $ :"
    index = 0
    for tag in a.split():
        t_i_dict[tag] = index
        index = index + 1
    for k in t_i_dict.keys():
        value = t_i_dict[k]
        i_t_dict[value] = k
    WordTag = {}
    WordTagIndex = {}
    for word in WordTagProb.keys():
        poss = []
        tag_vec_list = []
        for tag in WordTagProb[word].keys():
            poss.append(float(WordTagProb[word][tag]))
            tag_vec_list.append(tag)
        tag = np.random.choice(tag_vec_list, p = np.array(poss).ravel())
        if word in w_i_dict.keys():
            WordTag[w_i_dict[word]] = tag
            WordTagIndex[w_i_dict[word]] = t_i_dict[tag]
    for word in WordTag.keys():
        WordTag[word] = TagVec.wv[WordTag[word]]
    WordTag[str(max(w_i_dict.values()))] = [float(0.0) for x in range(emb_dim)]
    return WordTag,WordTagIndex,i_t_dict

def GetTagProbList(i_t_dict,tag_bi):
    a = "CC CD DT EX FW IN JJ JJR JJS LS MD NN NNS NNP NNPS PDT POS PRP PRP$ RB RBR RBS RP SYM TO UH VB VBD VBG VBN VBP VBZ WDT WP WP$ WRB , $ :"
    TagProbList = []
    for i in range(len(a.split())):
        former = i_t_dict[i]
        for j in range(len(a.split())):
            later = i_t_dict[j]
            bigram = str(former)+" "+str(later)
            TagProbList.append(tag_bi[bigram])
    return TagProbList 

def generate_samples_gen(sess, trainable_model, batch_size, generated_num, iw_dict, wi_dict, NoLike, Like, output_file=None, train=0, real_file=None):
    # Generate Samples
    WordVec = gensim.models.Word2Vec.load(sys.path[0] + r'/Preparation/save/wordvec_model')
    generated_samples = []
    for _ in range(int(64 / batch_size)): #you must change the number here, which is equal to generated_num
        generated_samples.extend(trainable_model.generate(sess, 1.0, train))
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer_new = []
                for x in poem:
                    if x == '\n':
                        buffer_new.append("\n")
                    else:
                        if str(x) in iw_dict.keys():
                            if iw_dict[str(x)] in NoLike:
                                flag = True
                                if iw_dict[str(x)] in WordVec.wv.vocab.items():
                                    for i in WordVec.wv.most_similar(iw_dict[str(x)]):
                                        if wi_dict[i[0]] in Like:
                                            flag = False
                                            buffer_new.append(wi_dict[i[0]])
                                            print("alter---"+str(x)+"---with---"+wi_dict[i[0]])
                                            break
                                    if flag:
                                        buffer_new.append(str(x))
                                else:
                                    buffer_new.append(str(x))
                            else:
                                buffer_new.append(str(x))
                        else:
                            buffer_new.append(str(x))
                buffer_last = ' '.join(buffer_new) + '\n'
                fout.write(buffer_last)
    with open(output_file,'r') as Reader, open(real_file+str(time()),'w') as Writer:
        for index,line in enumerate(Reader):
            word_index = line.split()
            words = []
            for i in word_index:
                if i in iw_dict.keys():
                    words.append(iw_dict[i])
            Writer.write(" ".join(words)+'\n')

class Persongan(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_boolean('restore', False, 'Training or testing a model')
        flags.DEFINE_boolean('resD', False, 'Training or testing a D model')
        flags.DEFINE_integer('length', 20, 'The length of toy data')
        flags.DEFINE_string('model', "", 'Model NAME')
        self.sequence_length = FLAGS.length
        self.filter_size = [2, 3]
        self.num_filters = [125,125] #sum the filter
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 32
        self.generate_num = 64
        self.start_token = 0
        self.dis_embedding_dim = 75  #the dimension of one word vector
        self.goal_size = 16
         
        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'
        self.test_file = 'save/test_file.txt'
        self.generate_real_file = 'save/final.txt'

    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

    def train_discriminator(self,iw_dict, wi_dict,Nolike, Like):
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, iw_dict,wi_dict,Nolike, Like,self.generator_file,real_file=self.generate_real_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file) #oracle is real text(positive sample) index
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()#x is sentence, y is label
            feed = {
                self.discriminator.D_input_x: x_batch,
                self.discriminator.D_input_y: y_batch,
            }
            _, _ = self.sess.run([self.discriminator.D_loss, self.discriminator.D_train_op,], feed)
            self.generator.update_feature_function(self.discriminator) #extract new feature 

    def evaluate(self,iw_dict,wi_dict,Nolike, Like):
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, iw_dict,wi_dict,Nolike, Like,self.generator_file,real_file=self.generate_real_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super().evaluate()

    def init_real_trainng(self, data_loc=None,author_name=None, StrucWord=None,WordTagProb=None,TagBi=None): 
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        self.sequence_length, self.vocab_size = text_precess(data_loc)
        tokens = get_tokenlized(data_loc)
        word_set = get_word_list(tokens)
        [word_index_dict, index_word_dict] = get_dict(word_set)
        LikeWord = LikeWordIndex(word_index_dict,author_name)
        NoLikeWord = NoLikeWordIndex(word_index_dict,author_name)
        StrucWordList = GetStructIndex(word_index_dict,StrucWord)
        WordTag,WordTagIndex,i_t_dict = GetWordTag(word_index_dict,WordTagProb,self.emb_dim)
        TagProbList = GetTagProbList(i_t_dict,TagBi)
        MaxIndex = max(word_index_dict.values())
        goal_out_size = sum(self.num_filters)+2*self.dis_embedding_dim 
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      dis_emb_dim=self.dis_embedding_dim, filter_sizes=self.filter_size,
                                      num_filters=self.num_filters,
                                      batch_size=self.batch_size, hidden_dim=self.hidden_dim,
                                      start_token=self.start_token,
                                      goal_out_size=goal_out_size, step_size=4,
                                      l2_reg_lambda=self.l2_reg_lambda, LikeWord=LikeWord, 
                                      StrucWord=StrucWordList,word_tag_index=WordTagIndex,tag_prob_list=TagProbList)
        self.set_discriminator(discriminator)
        generator = Generator(num_classes=2, num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                              emb_dim=self.emb_dim, dis_emb_dim=self.dis_embedding_dim, goal_size=self.goal_size,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              filter_sizes=self.filter_size, start_token=self.start_token,
                              num_filters=self.num_filters, goal_out_size=goal_out_size, D_model=discriminator,
                              step_size=4,iw_dict=index_word_dict,word_tag=WordTag,max_index=MaxIndex)
        self.set_generator(generator)
        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
        return word_index_dict, index_word_dict,LikeWord,NoLikeWord

    def init_real_metric(self):
        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

    def train_real(self, data_loc=None, author_name=None,StrucWord = None, WordTagPro = None, TagBi=None):
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized
        wi_dict, iw_dict,Like,Nolike = self.init_real_trainng(data_loc,author_name,StrucWord,WordTagPro,TagBi)
        self.init_real_metric()
        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))
        self.sess.run(tf.global_variables_initializer())
        self.pre_epoch_num = 10 #change here
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-persongan-real.csv', 'w')
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, iw_dict,wi_dict,
                             Nolike, Like,self.generator_file,real_file=self.generate_real_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        for a in range(1):
            g = self.sess.run(self.generator.gen_x, feed_dict={self.generator.drop_out: 1, self.generator.train: 1})

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, iw_dict,
                                     wi_dict,Nolike, Like,self.generator_file,real_file=self.generate_real_file)
                get_real_test_file()
                self.evaluate(iw_dict,wi_dict,Nolike, Like)

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator(iw_dict,wi_dict,Nolike,Like)


        self.reset_epoch()
        self.reward = Reward(model=self.generator, dis=self.discriminator, sess=self.sess, rollout_num=4)
        for epoch in range(self.adversarial_epoch_num//10): #adversarial_epoch_num is real train(after pre-train); 
            for epoch_ in range(10): #every 10 epoch train 
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                start = time()
                for index in range(1): #1 time train generator
                    samples = self.generator.generate(self.sess, 1) # format[[sentence 1],[sentence 2],[],[]] 64 ge sentences
                    rewards = self.reward.get_reward(samples)
                    feed = {
                        self.generator.x: samples,
                        self.generator.reward: rewards,
                        self.generator.drop_out: 1
                    }
                    _, _, g_loss, w_loss = self.sess.run(
                        [self.generator.manager_updates, self.generator.worker_updates, self.generator.goal_loss,
                         self.generator.worker_loss, ], feed_dict=feed)
                    print('epoch', str(epoch), 'goal_loss', g_loss, 'worker_loss', w_loss)
                end = time()
                self.add_epoch()
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, iw_dict,
                                         wi_dict,Nolike, Like,self.generator_file,real_file=self.generate_real_file)
                    get_real_test_file()
                    self.evaluate(iw_dict,wi_dict,Nolike, Like)

                for _ in range(15): #15 time train discriminator
                    self.train_discriminator(iw_dict,wi_dict,Nolike, Like)
            for epoch_ in range(5): #5 times MLE train generator
                start = time()
                loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
                end = time()
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                if epoch % 5 == 0:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,iw_dict,
                                         wi_dict,Nolike, Like,self.generator_file,real_file=self.generate_real_file)
                    get_real_test_file()
                    # self.evaluate()
            for epoch_ in range(5): #5 times discriminator train MLE
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                self.train_discriminator(iw_dict,wi_dict,Nolike, Like)

