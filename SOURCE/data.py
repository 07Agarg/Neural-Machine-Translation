# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 05:29:16 2018

@author: ashima.garg
"""
import os 
import config

class Data():
    def __init__(self, src_vocab, tgt_vocab):
        self.src_vocab_len = None
        self.tgt_vocab_len = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.src_sent = []
        self.tgt.sent = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def format_sent(self, is_source, sents):
        for line in sents:
            line = line.split(' ')
            for i, word in line:
                if is_source:
                    if word not in self.src_vocab.keys():
                        line[i] = '<unk>'
                else:
                    if word not in self.tgt_vocab.keys():
                        line[i] = '<unk>'
            if is_source:
                self.src_sent.append(line)
            else:
                self.tgt_sent.append(line)
                        
                    
    def load_data(self, folder_path, file_path, is_source):
        filepath = os.path.join(folder_path, file_name)
        sents = []
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                sents.append(line)
                if len(sents) >= 100000:
                    break
        self.format_sent(is_source, sents)
        
    def generate_batch(self):
        
