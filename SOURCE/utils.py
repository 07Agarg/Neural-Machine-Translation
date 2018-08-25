# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 04:43:22 2018

@author: ashima.garg
"""
import os
import config

def load_dict(self, dictpath, is_source):
    filepath = os.path.join(config.VOCAB_DIR, dictpath)
    vocab = dict()
    with open(filepath, "r", encoding="utf-8") as f:
        for word in f:
            vocab[word[:-1]] = len(vocab)
    return vocab
    

