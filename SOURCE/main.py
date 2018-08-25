# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 05:29:16 2018

@author: ashima.garg
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import data
import model
import config
import utils
import tensorflow as tf


if __name__ == "__main__":
    with tf.Graph().as_default():
        src_vocab = utils.load_dict(config.vocab_en, True)
        tgt_vocab = utils.load_dict(config.vocab_vi, False)
        
        #Train Data
        train_data = data.Data(src_vocab, tgt_vocab)
        train_data.load_data(config.TRAIN_EN_VI, config.train_en, True)
        train_data.load_data(config.TRAIN_EN_VI, config.train_vi, False)
        
        #Validation Data
        test_data = data.Data(src_vocab, tgt_vocab)
        test_data.load_data(config.VALIDATION_EN_VI, config.dev_en, True)
        test_data.load_data(config.VALIDATION_EN_VI, config.dev_vi, False)
        
        #Test Data
        val_data = data.Data(src_vocab, tgt_vocab)
        val_data.load_data(config.TEST_EN_VI, config.test_en, True)
        val_data.load_data(config.TEST_EN_VI, config.test_vi, False)
        
        
        