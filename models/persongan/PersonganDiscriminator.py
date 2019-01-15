import tensorflow as tf
import  numpy as np

def cosine_similarity(a,b):
    normalize_a = tf.nn.l2_normalize(a, -1)
    normalize_b = tf.nn.l2_normalize(b, -1)
    cos_similarity = (tf.multiply(normalize_a, normalize_b))
    return cos_similarity

# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output
    return output

class Discriminator(object):
    def __init__(self, sequence_length, num_classes, vocab_size,dis_emb_dim,filter_sizes, num_filters,batch_size,hidden_dim,
                 start_token,goal_out_size,step_size,LikeWord, StrucWord,word_tag_index,tag_prob_list,l2_reg_lambda=0.0, dropout_keep_prob=0.75 ):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.l2_reg_lambda = l2_reg_lambda
        self.num_filters_total = sum(self.num_filters)
        self.temperature = 1.0
        self.grad_clip = 5.0
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.dropout_keep_prob = dropout_keep_prob
        self.LikeWord = LikeWord
        self.StrucWord = StrucWord
        self.D_input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.D_input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.word_tag_index = word_tag_index
        self.tag_prob_list = tag_prob_list
        self.tag_len = 39
        self.max_word = max([int(x) for x in word_tag_index.keys()])
        self.WordTagIndexArray = tf.TensorArray(dtype=tf.int32, size=0,
                                                          dynamic_size=True, infer_shape=True, clear_after_read=False)
        self.TagProbArray = tf.TensorArray(dtype=tf.float32, size=0,
                                                          dynamic_size=True, infer_shape=True, clear_after_read=False)   
        
        for i in range(len(tag_prob_list)):
            self.TagProbArray = self.TagProbArray.write(i,self.tag_prob_list[i])
        for word in self.word_tag_index.keys():
            self.WordTagIndexArray = self.WordTagIndexArray.write(int(word),self.word_tag_index[word])
            
        with tf.name_scope('D_update'):
            self.D_l2_loss = tf.constant(0.0)
            self.FeatureExtractor_unit = self.FeatureExtractor()

            # Train for Discriminator
            with tf.variable_scope("feature") as self.feature_scope:
                D_feature = self.FeatureExtractor_unit(self.D_input_x,self.dropout_keep_prob)#,self.dropout_keep_prob)
                self.feature_scope.reuse_variables()

            D_scores, D_predictions,self.ypred_for_auc = self.classification(D_feature)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=D_scores, labels=self.D_input_y)
            self.D_loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.D_l2_loss

            self.D_params = [param for param in tf.trainable_variables() if
                             'Discriminator' or 'FeatureExtractor' in param.name]
            d_optimizer = tf.train.AdamOptimizer(5e-5)
            D_grads_and_vars = d_optimizer.compute_gradients(self.D_loss, self.D_params, aggregation_method=2)
            self.D_train_op = d_optimizer.apply_gradients(D_grads_and_vars)
    
    # This module used to Extract sentence's Feature
    def FeatureExtractor(self):
        def unit(Feature_input,dropout_keep_prob):#,dropout_keep_prob):
            with tf.variable_scope('FeatureExtractor') as scope:
                with tf.device('/cpu:0'), tf.name_scope("embedding") as scope:
                    W_fe = tf.get_variable(
                        name="W_fe",
                        initializer=tf.random_uniform([self.vocab_size + 1, self.dis_emb_dim], -1.0, 1.0)) #word embedding random initial
                    self.LikeWord = [int(x) for x in self.LikeWord]
                    self.LikeWordEmb = tf.nn.embedding_lookup(W_fe,self.LikeWord)
                    self.StrucWord = [int(x) for x in self.StrucWord]
                    self.StrucWordEmb = tf.nn.embedding_lookup(W_fe,self.StrucWord)
                    self.TempSeqLen = self.sequence_length
                    self.TempSeqLen_ = self.sequence_length
                    self.MaxWord = self.max_word
                    self.TagLen = self.tag_len
                    self.DisEmbDim = self.dis_emb_dim
                    def high_fn(feature):
                        temp,word_vec,flag = tf.map_fn(low_fn,feature,dtype=(tf.float32,tf.float32,tf.int32))
                        def cond_bi(i,Temp,feature,temp,OldTag,OldProb,NewTag,NewProb,const,t):
                            return i<self.TempSeqLen_ -1
                        def body_bi(i,Temp,feature,temp,OldTag,OldProb,NewTag,NewProb,const,t):
                            def false_bi(feature_,OldTag,const):
                                def true_low_bi(OldTag):
                                    return const,OldTag
                                def false_low_bi(OldTag,feature_):
                                    NewTag = self.WordTagIndexArray.read(feature_)
                                    return self.TagProbArray.read(tf.to_int32(tf.multiply(OldTag,self.TagLen)+NewTag)),NewTag
                                NewProb,NewTag = tf.cond(tf.less(feature_,tf.constant(0)),lambda: true_low_bi(OldTag),
                                                  lambda: false_low_bi(OldTag,feature_))
                                return NewProb,NewTag
                            def true_up_bi(OldTag,const):
                                return const,OldTag
                            def last_tag(OldTag,OldProb):
                                return OldProb,OldTag
                            
                            def not_last_tag(f,OldTag,const):
                                NewProb,NewTag = tf.cond(tf.greater(f,self.MaxWord),lambda:true_up_bi(OldTag,const),
                                        lambda: false_bi(f,OldTag,const))
                                return NewProb,NewTag 
                            i = i + 1
                            NewProb,NewTag = tf.cond(tf.equal(i,self.TempSeqLen_),lambda: last_tag(OldTag,OldProb),lambda:not_last_tag(feature[i],OldTag,const))
                            Temp = tf.concat([Temp,tf.expand_dims(tf.multiply(NewProb,temp[i-1]),0)],0)
                            return i,Temp,feature,temp,NewTag,NewProb,NewTag,NewProb,const,t
                        k = tf.constant(0)
                        k_ = tf.constant(0.0)
                        _,Newtemp,_,_,_,_,_,_,_,_ = tf.while_loop(cond_bi,body_bi,[k,tf.zeros([1,self.DisEmbDim]),feature,temp,self.WordTagIndexArray.read(0),k_,self.WordTagIndexArray.read(0),k_,k_,temp[0]],
                                                       shape_invariants=[k.get_shape(),tf.TensorShape([None,self.DisEmbDim]),feature.get_shape(),temp.get_shape(),k.get_shape(),k.get_shape(),k.get_shape(),k.get_shape(),k.get_shape(),tf.TensorShape(None)])
                        def body_vec(i,const,threhold,word_vec,vec):
                            def true_vec(i,j):
                                return i,j
                            def false_vec(i,j):
                                return i,j
                            threhold,vec = tf.cond(tf.equal(tf.reduce_mean(word_vec),const),lambda: true_vec(threhold,vec),
                                                   lambda: false_vec(i,word_vec[i]))              
                            i = i + 1
                            return i,const,threhold,word_vec,vec
                        def cond_vec(i,const,threhold,word_vec,vec):
                            return i<threhold-2
                        const = tf.Variable(lambda: tf.constant(0.0,dtype=tf.float32))
                        i,_,self.TempSeqLen,word_vec,vec = tf.while_loop(cond_vec,body_vec,[0,const,self.TempSeqLen,word_vec,tf.zeros(self.dis_emb_dim)])
                        return Newtemp,word_vec,flag,vec
                    def low_fn(elem):
                        flag = 0
                        key_word_vec = tf.convert_to_tensor([0.0 for x in range(self.dis_emb_dim)])
                        temp = tf.to_float(tf.nn.embedding_lookup(W_fe, elem+1))
                        def cond_key(i,threhold,LikeWordEmb,elem,key_word_vec):
                            return i<threhold
                        def body_key(i,threhold,LikeWordEmb,elem,key_word_vec):
                            def true_key(threhold,j):
                                return threhold,j
                            def false_key(i,key_word_vec):
                                return i,key_word_vec
                            threhold,key_word_vec = tf.cond(tf.equal(tf.reduce_mean(elem-LikeWordEmb[i]),tf.constant(0.0)),lambda: true_key(threhold,LikeWordEmb[i]),
                                                   lambda: false_key(i,key_word_vec))
                            i = i + 1
                            return i,threhold,LikeWordEmb,elem,key_word_vec
                        _,_,self.LikeWordEmb,temp,key_word_vec = tf.while_loop(cond_key,body_key,[0,len(self.LikeWord),self.LikeWordEmb,temp,key_word_vec])
                        def cond_stru(i,threhold,flag,strucword,elem):
                            return i<threhold
                        def body_stru(i,threhold,flag,strucword,elem):
                            def true_stru(i):
                                flag = 1
                                threhold = i
                                return flag,threhold
                            def false_stru(threhold):
                                flag = 0
                                threhold_ = threhold
                                return flag,threhold_
                            flag, threhold = tf.cond(tf.equal(tf.reduce_mean(temp-strucword[i]),tf.constant(0.0)),
                                                    lambda: true_stru(i),
                                                    lambda: false_stru(threhold))
                            i = i + 1
                            return i,threhold,flag,strucword,elem
                        _,_,flag,_,_= tf.while_loop(cond_stru,body_stru,[0,len(self.StrucWord),flag,self.StrucWordEmb,temp])                        
                        flag = tf.convert_to_tensor(flag)
                        return temp,key_word_vec,flag 
                    embedded_chars,wordVec,FlagSet,vec = tf.map_fn(high_fn,Feature_input,dtype=(tf.float32,tf.float32,tf.int32,tf.float32))
                    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
                # Create a convolution + maxpool layer for each filter size
                pooled_outputs = []
                for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                    with tf.name_scope("conv-maxpool-%s" % filter_size) as scope:
                        # Convolution Layer
                        filter_shape = [filter_size, self.dis_emb_dim, 1, num_filter]
                        W = tf.get_variable(name="W-%s" % filter_size,
                                            initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                        b = tf.get_variable(name="b-%s" % filter_size,
                                            initializer=tf.constant(0.1, shape=[num_filter]))
                        conv = tf.nn.conv2d(
                            embedded_chars_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv-%s" % filter_size)
                        # Apply nonlinearityprint(sess.run(i))
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-%s" % filter_size)
                        # Maxpooling over the outputs
                        def cond(i,FlagSet,h,pooleds,wordvec):
                            return i<self.batch_size-1
                        
                        def body(i,FlagSet,h,pooleds,wordvec):
                            sent_flag = tf.nn.embedding_lookup(FlagSet,i)
                            j = tf.arg_max(sent_flag,0)
                            pool = tf.cond(tf.less(j,tf.to_int64(tf.constant(1))),lambda: false_fn(i,h,wordvec), lambda: true_fn(i,j,h,wordvec))
                            if i==0:
                                pooleds = pool
                            else:
                                pooleds = tf.concat([pooleds,pool],0)
                            i = i + 1
                            return i,FlagSet,h,pooleds,wordvec
                        
                        def true_fn(i,j,h,wordvec):
                            temp = tf.expand_dims(h[i],0) 
                            def up_fn(temp,filter_size,wordvec):
                                pool = tf.nn.max_pool(temp, 
                                                      ksize=[1,self.sequence_length - filter_size + 1, 1, 1],
                                                      strides = [1,1,1,1],
                                                      padding='VALID')
                                pool = tf.concat([pool,tf.expand_dims(tf.expand_dims(tf.expand_dims(wordvec[i],0),0),0)],3)
                                return pool
                            def down_fn(h,wordvec):
                                pool = h[i][j]
                                pool = tf.concat([tf.expand_dims(tf.expand_dims(pool,0),0),tf.expand_dims(tf.expand_dims(tf.expand_dims(wordvec[i],0),0),0)],3)
                                return pool
                            
                            pool = tf.cond(tf.greater_equal(j,tf.constant(self.sequence_length-2,dtype=tf.int64)),
                                           lambda:  up_fn(temp,filter_size,wordvec) ,
                                           lambda: down_fn(h,wordvec))
                            return pool
                        
                        def false_fn(i,h,wordvec):
                            temp = tf.expand_dims(h[i],0)
                            pool = tf.nn.max_pool(temp, 
                                                  ksize=[1,self.sequence_length - filter_size + 1, 1, 1],
                                                  strides = [1,1,1,1],
                                                  padding='VALID')
                            pool = tf.concat([pool,tf.expand_dims(tf.expand_dims(tf.expand_dims(wordvec[i],0),0),0)],3)
                            return pool
                        
                        pooleds = tf.ones([1,1,1,sum(self.num_filters)/2+self.dis_emb_dim])
                        i = tf.constant(0)
                        i,FlagSet,h,pooleds,vec = tf.while_loop(cond,body,[i,FlagSet,h,pooleds,vec],shape_invariants=[i.get_shape(),
                                                                                                              FlagSet.get_shape(),
                                                                                                              h.get_shape(),
                                                                                                              tf.TensorShape([None,None,1,sum(self.num_filters)/2+self.dis_emb_dim]),
                                                                                                              vec.get_shape()])
                        pooled_outputs.append(pooleds)
                # Combine all the pooled features  
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total+2*self.dis_emb_dim])
                # Add highway
                with tf.name_scope("highway"):
                    h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)
                    # Add dropout
                with tf.name_scope("dropout"):
                    h_drop = tf.nn.dropout(h_highway,dropout_keep_prob)
            return h_drop
        return unit

    def classification(self, D_input):
        with tf.variable_scope('Discriminator'):
            W_d = tf.Variable(tf.truncated_normal([self.num_filters_total+2*self.dis_emb_dim, self.num_classes], stddev=0.1), name="W")
            b_d = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.D_l2_loss += tf.nn.l2_loss(W_d)
            self.D_l2_loss += tf.nn.l2_loss(b_d)
            self.scores = tf.nn.xw_plus_b(D_input, W_d, b_d, name="scores")
            self.ypred_for_auc = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        return self.scores, self.predictions, self.ypred_for_auc

