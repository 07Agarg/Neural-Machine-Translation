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
    def __init__(self, config):
        self.config = config
        self.src_inputs = tf.placeholder(shape=[self.config.src_seq_len, self.config.batch_size], dtype=tf.int32)
        self.tgt_inputs = tf.placeholder(shape=[self.config.tgt_seq_len, self.config.batch_size], dtype=tf.int32)
        self.tgt_labels = tf.placeholder(shape=[self.config.tgt_seq_len, self.config.batch_size], dtype=tf.int32)
        self.loss = None
        self.logits = None
        self.encoder_initial_state = None
        self.encoder_final_state = None
        self.decoder_initial_state = None
        self.decoder_final_state = None


    def build_encoder(self):
        embedding_layer = neural_network.Embedding_Layer([self.config.vocab_size, self.config.num_units])
        encoder_emb_inputs = embedding_layer.lookup(self.src_inputs)
        
        rnn_graph = neural_network.RNN_Graph([self.config.hidden_size, self.config.num_layers], self.training, self.config.keep_prob, self.config.batch_size)
        self.encoder_initial_state = rnn_graph.initial_state
        output, state = rnn_graph.encoder_feed_forward(encoder_emb_inputs, self.config)
        self.encoder_final_state = state

    def build_decoder(self):
        embedding_layer = neural_network.Embedding_Layer([self.config.vocab_size, self.config.num_units])
        decoder_emb_inputs = embedding_layer.lookup(self.tgt_inputs)
        
        rnn_graph = neural_network.RNN_Graph([self.config.hidden_size, self.config.num_layers], self.training, self.config.keep_prob, self.config.batch_size)
        self.decoder_initial_state = self.encoder_final_state
        
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, [config.tgt_max_sent_length-1 for _ in range(config.batch_size)], time_major = True)
        projection_layer = layers_core.Dense(config.tgt_vocab_size, use_bias=False, name="output_projection")
        decoder = tf.contrib.seq2seq.BasicDecoder(rnn_graph.model_cell, helper, self.decoder_initial_state,output_layer = projection_layer)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, swap_memory=True)
        self.logits = outputs.rnn_output
        

    def loss(self):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tgt_labels, logits=self.logits)
        self.loss = (tf.reduce_sum(crossent * target_weights) /(config.batch_size * self.config.num_units))
        
    def train(self):
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        grads, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(0.0001)
        update_step = optimizer.apply_gradients(zip(grads, params))
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            state = session.run(self.encoder_initial_state)
            for epoch in range(self.config.num_train_steps):
                avg_cost = 0
                