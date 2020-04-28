#encoding:utf-8
import tensorflow as tf
import numpy as np
import cPickle as pickle

from utils import layers 
import utils.operations as op
from keras.layers import Conv1D,GlobalMaxPooling1D,Dense

class Net(object):
    '''Add positional encoding(initializer lambda is 0),
       cross-attention, cnn integrated and grad clip by value.

    Attributes:
        conf: a configuration paramaters dict
        word_embedding_init: a 2-d array with shape [vocab_size+1, emb_size]
    '''
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        #config 构造head_nums
        self._multihead = layers.MultiHeadAttention(conf['emb_size'],conf['head_nums'])
        self._conv1d = Conv1D(filters=conf['emb_size'], kernel_size=5, activation='relu')
        self._pool1d = GlobalMaxPooling1D()
        self._dense1 = Dense(conf['emb_size'])
        self._dense2 = Dense(conf['emb_size'])

        if self._conf['word_emb_init'] is not None:
            print('loading word emb init')
            self._word_embedding_init = pickle.load(open(self._conf['word_emb_init'], 'rb'))
        else:
            self._word_embedding_init = None

    def build_graph(self):
        with self._graph.as_default():
            if self._conf['rand_seed'] is not None:
                rand_seed = self._conf['rand_seed']
                tf.set_random_seed(rand_seed)
                print('set tf random seed: %s' %self._conf['rand_seed'])

            #word embedding
            if self._word_embedding_init is not None:
                word_embedding_initializer = tf.constant_initializer(self._word_embedding_init)
            else:
                word_embedding_initializer = tf.random_normal_initializer(stddev=0.1)

            self._word_embedding = tf.get_variable(
                name='word_embedding',
                shape=[self._conf['vocab_size']+1, self._conf['emb_size']],
                dtype=tf.float32,
                initializer=word_embedding_initializer)


            #define placehloders
            #config max_turn_history_num
            self.turns_history = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_history_num"], self._conf["max_turn_len"]])
            
            self.turns = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_num"], self._conf["max_turn_len"]])

            self.tt_turns_len = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"]])

            self.every_turn_len = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_num"]])
    
            self.response = tf.placeholder(
                tf.int32, 
                shape=[self._conf["batch_size"], self._conf["max_turn_len"]])

            self.response_len = tf.placeholder(
                tf.int32, 
                shape=[self._conf["batch_size"]])

            self.label = tf.placeholder(
                tf.float32, 
                shape=[self._conf["batch_size"]])


            #define operations
            #response part
            Hr = tf.nn.embedding_lookup(self._word_embedding, self.response)
            turns_history_embedding = tf.nn.embedding_lookup(self._word_embedding, self.turns_history)
            

            if self._conf['is_positional'] and self._conf['stack_num'] > 0:
                with tf.variable_scope('positional'):
                    Hr = op.positional_encoding_vector(Hr, max_timescale=10)
            Hr_stack = [Hr] 
            
            _batch_size ,_turn_nums, _turn_words, _emb_size = turns_history_embedding.get_shape().as_list()
            turns_history_embedding = tf.reshape(turns_history_embedding, [-1,_turn_words,_emb_size])
            
            for index in range(self._conf['stack_num']):
                turns_history_embedding, _ =self._multihead(turns_history_embedding,
                                                         turns_history_embedding,
                                                         turns_history_embedding)
            
            turns_history_embedding = tf.reshape(turns_history_embedding, 
                                                 [_batch_size ,_turn_nums, _turn_words, _emb_size])
           

            for index in range(self._conf['stack_num']):
                with tf.variable_scope('self_stack_' + str(index)):
                    Hr = layers.block(
                        Hr, Hr, Hr, 
                        Q_lengths=self.response_len, K_lengths=self.response_len)
                    Hr_stack.append(Hr)
                    
            with tf.variable_scope('respone_extraction_history'):
                turn_important_inf = []
                #需要增加一个全链接层
                for _t in tf.split(turns_history_embedding, self._conf['max_turn_history_num'],1):
                    _t = tf.squeeze(_t)
                    #_match_result = layers.attention(Hr_stack[-1], _t,  _t, self.response_len, self.response_len)
                    _match_result = layers.attention( self._dense1(Hr_stack[-1]), _t,  _t, self.response_len, self.response_len)
                    turn_important_inf.append(tf.expand_dims(_match_result,1))
            
            best_turn_match = tf.concat(turn_important_inf,1)
            with tf.variable_scope('response_extraciton_best_information'):
                #best_information,_ = self._multihead(Hr_stack[-1], best_turn_match, best_turn_match)
                best_information,_ = self._multihead(self._dense2(Hr_stack[-1]), best_turn_match, best_turn_match)
                best_information = layers.FFN(best_information)
                

            #context part
            #a list of length max_turn_num, every element is a tensor with shape [batch, max_turn_len]
            list_turn_t = tf.unstack(self.turns, axis=1) 
            list_turn_length = tf.unstack(self.every_turn_len, axis=1)
            
            sim_turns = []
            #for every turn_t calculate matching vector
            for turn_t, t_turn_length in zip(list_turn_t, list_turn_length):
                Hu = tf.nn.embedding_lookup(self._word_embedding, turn_t) #[batch, max_turn_len, emb_size]

                if self._conf['is_positional'] and self._conf['stack_num'] > 0:
                    with tf.variable_scope('positional', reuse=True):
                        Hu = op.positional_encoding_vector(Hu, max_timescale=10)
                Hu_stack = [Hu]

                for index in range(self._conf['stack_num']):

                    with tf.variable_scope('self_stack_' + str(index), reuse=True):
                        Hu = layers.block(
                            Hu, Hu, Hu,
                            Q_lengths=t_turn_length, K_lengths=t_turn_length)

                        Hu_stack.append(Hu)



                r_a_t_stack = []
                t_a_r_stack = []
                for index in range(self._conf['stack_num']+1):

                    with tf.variable_scope('t_attend_r_' + str(index)):
                        try:
                            t_a_r = layers.block(
                                tf.add(Hu_stack[index],best_information), Hr_stack[index], Hr_stack[index],
                                Q_lengths=t_turn_length, K_lengths=self.response_len)
                        except ValueError:
                            tf.get_variable_scope().reuse_variables()
                            t_a_r = layers.block(
                                tf.add(Hu_stack[index],best_information), Hr_stack[index], Hr_stack[index],
                                Q_lengths=t_turn_length, K_lengths=self.response_len)


                    with tf.variable_scope('r_attend_t_' + str(index)):
                        try:
                            r_a_t = layers.block(
                                Hr_stack[index],
                                tf.add(Hu_stack[index],best_information), 
                                tf.add(Hu_stack[index],best_information),
                                Q_lengths=self.response_len, K_lengths=t_turn_length)
                        except ValueError:
                            tf.get_variable_scope().reuse_variables()
                            r_a_t = layers.block(
                                Hr_stack[index], 
                                tf.add(Hu_stack[index],best_information), 
                                tf.add(Hu_stack[index],best_information),
                                Q_lengths=self.response_len, K_lengths=t_turn_length)

                    t_a_r_stack.append(t_a_r)
                    r_a_t_stack.append(r_a_t)

                t_a_r_stack.extend(Hu_stack)
                r_a_t_stack.extend(Hr_stack)
                
                t_a_r = tf.stack(t_a_r_stack, axis=-1)
                r_a_t = tf.stack(r_a_t_stack, axis=-1)

                            
                #calculate similarity matrix
                with tf.variable_scope('similarity'):
                    # sim shape [batch, max_turn_len, max_turn_len, 2*stack_num+1]
                    # divide sqrt(200) to prevent gradient explosion
                    sim = tf.einsum('biks,bjks->bijs', t_a_r, r_a_t) / tf.sqrt(200.0)

                sim_turns.append(sim)


            #cnn and aggregation
            sim = tf.stack(sim_turns, axis=1)
            print('sim shape: %s' %sim.shape)
            with tf.variable_scope('cnn_aggregation'):
                final_info = layers.CNN_3d(sim, 32, 16)
                #final_info_dim = final_info.get_shape().as_list()[-1]
                #for douban
                #final_info = layers.CNN_3d(sim, 16, 16)
                #                 _x = self._conv1d(best_information)
                #                 _x = self._pool1d(_x)
                #final_info = tf.concat([final_info,best_information],-1)

            #loss and train
            with tf.variable_scope('loss'):
                self.loss, self.logits = layers.loss(final_info, self.label)

                self.global_step = tf.Variable(0, trainable=False)
                initial_learning_rate = self._conf['learning_rate']
                self.learning_rate = tf.train.exponential_decay(
                    initial_learning_rate,
                    global_step=self.global_step,
                    decay_steps=400,
                    decay_rate=0.9,
                    staircase=True)

                Optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.optimizer = Optimizer.minimize(
                    self.loss,
                    global_step=self.global_step)

                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver(max_to_keep = self._conf["max_to_keep"])
                self.all_variables = tf.global_variables() 
                self.all_operations = self._graph.get_operations()
                self.grads_and_vars = Optimizer.compute_gradients(self.loss)

                for grad, var in self.grads_and_vars:
                    if grad is None:
                        print(var)

                self.capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.grads_and_vars]
                self.g_updates = Optimizer.apply_gradients(
                    self.capped_gvs,
                    global_step=self.global_step)
    
        return self._graph

