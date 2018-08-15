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
import tensorflow as tf


if __name__ == "__main__":
    with tf.Graph().as_default():