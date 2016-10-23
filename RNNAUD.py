'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import time
import numpy as np
import mdl_data
import sys

GPUNUM = sys.argv[1]
FILEPATH = sys.argv[2]

with tf.device('/gpu:' + GPUNUM):
    #Source reference: https://github.com/aymericdamien/TensorFlow-Examples.git/input_data.py
    def dense_to_one_hot(labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    
    # Load data
    data = mdl_data.YLIMED('YLIMED_info.csv', FILEPATH + '/YLIMED150924/audio/mfcc20', FILEPATH + '/YLIMED150924/keyframe/fc7')
    X_aud_train = data.get_aud_X_train()
    y_train = data.get_y_train()
    Y_train = dense_to_one_hot(y_train)

    p = np.random.permutation(len(Y_train))
    X_aud_train = X_aud_train[p]
    Y_train = Y_train[p]
    
    # Parameters
    learning_rate = 0.001
    training_epochs = 20
    batch_size = 256
    display_step = 1

    # Network Parameters
    n_input = 100
    n_steps = 20 # input * steps = 2000
    n_hidden = 512 # hidden layer num of features
    n_classes = 10 # YLI_MED total classes (0-9 digits)

    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def RNN(x, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    pred = RNN(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        #Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(Y_train)/batch_size)
            #Loop oveer all batches
            for i in range(total_batch):
                batch_x, batch_y, finish = data.next_batch(X_aud_train, Y_train, batch_size, len(Y_train))
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((batch_size, n_steps, n_input))
                # Fit traning using batch data
                sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict = {x: batch_x, y: batch_y}) / total_batch
                #Shuffling
                if finish:
                    p = np.random.permutation(len(Y_train))
                    X_aud_train = X_aud_train[p]
                    Y_train = Y_train[p]
            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
        print "Optimization Finished!"

        # Test model
        correct = 0
        correct_prediction = tf.equal(tf.argmax(tf.reduce_mean(pred, 0, keep_dims=True), 1), tf.argmax(y, 1))
        Vid_test = data.get_vid_info('Test')
        for i in range(len(Vid_test)):
            batch_x_aud = data.get_vid_data(Vid_test[i], 'Aud')
            batch_x_aud = batch_x_aud.reshape((-1, n_steps, n_input))
            batch_y = np.asarray([int(Vid_test[i].split()[2])])
            batch_y = dense_to_one_hot(batch_y)
            
            if correct_prediction.eval({x: batch_x_aud, y: batch_y}):
                correct+=1        
        # Calculate accuracy
        print "Accuracy:",  float(correct)/float(len(Vid_test))
        print 'RNNAUDIO.py'