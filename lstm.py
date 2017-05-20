#Roughly based around : https://github.com/jpmcd/TensorflowSentiment/blob/master/tf_lstm.py

import tensorflow as tf
import numpy as np
from IPython import embed

class LSTM(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, l2_reg_lambda=0.0,num_hidden=100):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout

        
        l2_loss = tf.constant(0.0) # Keeping track of l2 regularization loss

        #1. EMBEDDING LAYER ################################################################
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        #2. LSTM LAYER ######################################################################
        self.lstm_cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
        #self.h_drop_exp = tf.expand_dims(self.h_drop,-1)
        self.lstm_out,self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,self.embedded_chars,dtype=tf.float32)
        #embed()

        val2 = tf.transpose(self.lstm_out, [1, 0, 2])
        last = tf.gather(val2, int(val2.get_shape()[0]) - 1) 

        out_weight = tf.Variable(tf.random_normal([num_hidden, num_classes]))
        out_bias = tf.Variable(tf.random_normal([num_classes]))

        with tf.name_scope("output"):
            #lstm_final_output = val[-1]
            #embed()
            self.scores = tf.nn.xw_plus_b(last, out_weight,out_bias, name="scores")
            self.predictions = tf.nn.softmax(self.scores, name="predictions")

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss")

        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1),tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"),name="accuracy")
        print "LOADED LSTM!"
