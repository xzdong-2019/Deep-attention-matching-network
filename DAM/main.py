#encoding:utf-8
import sys
import os
import time

# import c as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import models.net as net
import utils.douban_evaluation as eva
#for douban
#import utils.douban_evaluation as eva

import bin.train_and_evaluate as train
import bin.test_and_evaluate as test
import utils.utils as load_data

# configure

conf = {
    "data_path": "./data/douban/data.pkl",
    "save_path": "./output/douban/DAM_hireaction_version_2.0/",
    "word_emb_init": "./data/douban/word_embedding.pkl",
    # "data_path": "./data/ubuntu/data.pkl",
    # "save_path": "./output/ubuntu/DAM_new/",
    # "word_emb_init": "./data/ubuntu/word_embedding.pkl",
    #"init_model": None,  #should be set for test  "output/douban/temp/model.ckpt.18"

    "init_model": "output/douban/DAM_hireaction_version_2.0/model.ckpt.14",  #should be set for test  "output/douban/temp/model.ckpt.18"
    "rand_seed": None, 

    "drop_dense": None,
    "drop_attention": None,

    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,  

    "stack_num": 5,  
    "attention_type": "dot",

    "learning_rate": 1e-3,
    #"vocab_size": 434512, #for ubuntu
    "vocab_size": 172130, #for douban
    "emb_size": 200,
    "batch_size": 200, #200 for test

    "max_turn_num": 4,  
    "max_turn_len": 50, 

    "max_to_keep": 1,
    "num_scan_data": 2,
    #"_EOS_": 28270, #for ubuntu data
    "_EOS_": 1, #1 for douban data
    "final_n_class": 1,
    "max_turn_history_num":5,
    "head_nums":4,
}


model = net.Net(conf)
#train.train(conf, model)


#test and evaluation, init_model in conf should be set
test.test(conf, model)

