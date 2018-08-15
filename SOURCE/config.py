# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 05:29:16 2018

@author: ashima.garg
"""
import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('..')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL')
SOURCE_DIR = os.path.join(ROOT_DIR, 'SOURCE')
VOCAB_DIR = os.path.join(ROOT_DIR, 'VOCAB')

# DATA FILES
TRAIN_EN_VI = os.path.join(DATA_DIR, "train-en-vi")
VALIDATION_EN_VI = os.path.join(DATA_DIR,  "dev-2012-en-vi")
TEST_EN_VI = os.path.join(DATA_DIR, "test-2013-en-vi")

train_en = "train.en"
train_vi = "train.vi"
test_en = "tst2013.en"
test_vi = "tst2013.vi"
dev_en = "tst2012.en"
dev_vi = "tst2012.vi"

#Vocab_Files
vocab_en = "vocab.en"
vocab_vi = "vocab.vi"

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

num_units = 32
num_layers = 2
num_train_steps = 12000
src_max_len = 50
tgt_max_len = 50
batch_size = 128
dropout = 0.2

