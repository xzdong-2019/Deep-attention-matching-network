import sys
import os
import time

import cPickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva
with open('./data/douban/word2id', 'r') as fr:
    vocab_dict ={}
    for line in fr:
        con = line.strip().split()
        if len(con)==2:
            vocab_dict[con[0]]=con[1]

datas = []
with open('./data/douban/test.txt', 'r') as fr:
    for line in fr:
        # print(line)
        datas.append(line.strip())
max_len = 50

def generator():
    turns =  []
    respone = []
    respone_len = []
    every_turn_lens = []
    label = []
    def text_to_id(text,max_len):
        w_num = [0]*max_len
        length = len(text.split())
        if length >= 50:
            length = 50
        for i in range(len(text.split())):
            w = text.split()[i]
            if i == 50:
                break
            if w in vocab_dict:
                w_num[i]=vocab_dict[w]
            else:
                w_num[i] = 0
        return w_num, length
    for data in datas:
        var_context = data.split('\t')
        respone_num, respone_length = text_to_id(var_context[-1], max_len)
        respone.append(respone_num)
        respone_len.append(len(var_context[-1].split()))
        label.append(var_context[0])
        var_turn = []
        var_length = []
        for turn_text in var_context[1:len(var_context)-1]:
            num_text, length = text_to_id(turn_text, max_len)
            
            var_length.append(length)
            var_turn.append(num_text)
        if len(var_turn)>=9:
            var_turn=var_turn[:9]
            var_length = var_length[:9]
        else:
            for i in range(9-len(var_turn)):
                var_length.append(0)
                var_turn.append([0]*50)
        every_turn_lens.append(var_length)
        turns.append(var_turn)
    return turns, every_turn_lens, respone, respone_len, label

turns, every_turn_lens, respone, respone_len, label = generator()
def test(conf, _model):
    
    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # load data
    print('starting loading data')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train_data, val_data, test_data = pickle.load(open(conf["data_path"], 'rb'))    
    print('finish loading data')

    test_batches = reader.build_batches(test_data, conf)

    print("finish building test batches")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # refine conf
    test_batch_num = len(test_batches["response"])

    print('configurations: %s' %conf)


    _graph = _model.build_graph()
    print('build graph sucess')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    with tf.Session(graph=_graph) as sess:
        #_model.init.run();
        _model.saver.restore(sess, conf["init_model"])
        print("sucess init %s" %conf["init_model"])

        batch_index = 0
        step = 0

        score_file_path = conf['save_path'] + 'score.test'
        score_file = open(score_file_path, 'w')

        print('starting test')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        for i in xrange(len(label)): 
            turns, every_turn_lens, respone, respone_len, label
            feed = { 
                _model.turns: np.array([turns[i]]),
                _model.every_turn_len: np.array([every_turn_lens[i]]),
                _model.response: np.array([respone[i]]),
                _model.response_len: np.array([respone_len[i]]),
                _model.label: np.array([label[i]], dtype=np.float)
                }   
                
            scores = sess.run(_model.logits, feed_dict = feed)
            print(scores)
                    
#             for i in xrange(conf["batch_size"]):
#                 score_file.write(
#                     str(scores[i]) + '\t' + 
#                     str(test_batches["label"][batch_index][i]) + '\n')
#                     #str(sum(test_batches["every_turn_len"][batch_index][i]) / test_batches['tt_turns_len'][batch_index][i]) + '\t' 
#                     #str(test_batches['tt_turns_len'][batch_index][i]) + '\n') 

#         score_file.close()
#         print('finish test')
#         print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        
#         #write evaluation result
#         result = eva.evaluate(score_file_path)
#         result_file_path = conf["save_path"] + "result.test"
#         with open(result_file_path, 'w') as out_file:
#             for p_at in result:
#                 out_file.write(str(p_at) + '\n')
#         print('finish evaluation')
#         print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        

                    
