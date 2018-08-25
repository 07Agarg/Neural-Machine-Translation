# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 05:29:16 2018

@author: ashima.garg
"""

import tensorflow as tf 

class Embedding_Layer():
    
    def __init__(self, shape):
        self.embedding = tf.get_variable("embedding", shape=shape, dtype=tf.float32)
        
    def lookup(self, input_data):
        output = tf.nn.embedding_lookup(self.embedding, input_data)
        return output
    
class RNN_Graph():
    
    def __init__(self, shape, training, keep_prob, batch_size):
        
        def make_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(shape[0], forget_bias=0.0, state_is_tuple=True, reuse=not training)
            if training and keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob)
            return cell
        
        self.model_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(shape[1])], state_is_tuple=True)
        self.encoder_initial_state = self.model_cell.zero_state(batch_size, dtype=tf.float32)
        
    def encoder_feed_forward(self, input_data, config):
        output, state = tf.nn.dynamic_rnn(self.model_cell, input_data, initial_state = self.encoder_initial_state, sequence_length = source_seq_lengths, time_major=True)
        return output, state
    
        