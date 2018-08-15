# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 05:29:16 2018

@author: ashima.garg
"""
import tensorflow as tf
import os
import neural_network
from tensorflow.python.layers import core as layers_core
import config

class Model():
    def __init__(self):
        self.inputs = tf.placeholder(shape=[self.config.batch_size, self.config.num_steps], dtype=tf.int32)
        self.labels = tf.placeholder(shape=[self.config.batch_size, self.config.num_steps], dtype=tf.int32)
        self.loss = None
        self.logits = None
        self.encoder_initial_state = None
        self.encoder_final_state = None
        self.decoder_initial_state = None
        self.decoder_final_state = None


    def build_encoder(self):
        embedding_layer = neural_network.Embedding_Layer([self.config.vocab_size, self.config.hidden_size])
        encoder_emb_inputs = embedding_layer.lookup(self.inputs)
        
        rnn_graph = neural_network.RNN_Graph([self.config.hidden_size, self.config.num_layers], self.training, self.config.keep_prob, self.config.batch_size)
        self.encoder_initial_state = rnn_graph.initial_state
        output, state = rnn_graph.encoder_feed_forward(encoder_emb_inputs, self.config)
        self.encoder_final_state = state

    def build_decoder(self):
        embedding_layer = neural_network.Embedding_Layer([self.config.vocab_size, self.config.hidden_size])
        decoder_emb_inputs = embedding_layer.lookup(self.inputs)
        
        rnn_graph = neural_network.RNN_Graph([self.config.hidden_size, self.config.num_layers], self.training, self.config.keep_prob, self.config.batch_size)
        self.decoder_initial_state = self.encoder_final_state
        output, state = rnn_graph.decoder_feed_forward(inputs, self.config, self.decoder_final_state)
        self.decoder_final_state = state
        
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, [config.tgt_max_sent_length-1 for _ in range(config.batch_size)], time_major = True)
        self.output_layer = layers_core.Dense(config.tgt_vocab_size, use_bias=False, name="output_projection")
        decoder = tf.contrib.seq2seq.BasicDecoder(rnn_graph.model_cell, helper, self.decoder_initial_state,output_layer = self.output_layer)
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, swap_memory=True)
        self.logits = outputs.rnn_output

    def loss(self):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        self.loss = (tf.reduce_sum(crossent * target_weights) /config.batch_size)