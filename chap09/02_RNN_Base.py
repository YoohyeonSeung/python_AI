import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 1, 0]


with tf.variable_scope('one_cell') as scope:
    ## One Cell RNN input_dim(4) -> output_dim(2)
    hidden_size = 2
    cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_size)    # tensorflow에서는 추천하지 않음. pdf에선는 BasicLSTMCell을 씀
    print(cell.output_size, cell.state_size)