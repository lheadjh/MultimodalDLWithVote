#CUDA_VISIBLE_DEVICES=

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
    X_img_train = data.get_img_X_train()
    y_train = data.get_y_train()
    Y_train = dense_to_one_hot(y_train)

    # Shuffle initial data
    p = np.random.permutation(len(Y_train))
    X_img_train = X_img_train[p]
    Y_train = Y_train[p]

    learning_rate = 0.01
    training_epochs = 100
    batch_size = 256
    display_step = 1

    # Network Parameters
    n_input_img = 4096 # YLI_MED image data input (data shape: 4096, fc7 layer output)
    n_hidden = 1000 # 1st layer num features
    n_classes = 10 # YLI_MED total classes (0-9 digits)

    #image part

    x = tf.placeholder("float", [None, n_input_img])
    y = tf.placeholder("float", [None, n_classes])

    # Create model
    def multilayer_perceptron(_X, _weights, _biases):
        layer = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h']), _biases['b'])) #Hidden layer with RELU activation
        return tf.add(tf.matmul(layer, _weights['out']), _biases['out'])

    # Store layers weight & bias
    weights = {
        'h': tf.Variable(tf.random_normal([n_input_img, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'b': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graphe
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        #Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(Y_train)/batch_size)
            #Loop oveer all batches
            for i in range(total_batch):
                batch_xs, batch_ys, finish = data.next_batch(X_img_train, Y_train, batch_size, len(Y_train))
                # Fit traning using batch data
                sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict = {x: batch_xs, y: batch_ys}) / total_batch
                #Shuffling
                if finish:
                    p = np.random.permutation(len(Y_train))
                    X_img_train = X_img_train[p]
                    Y_train = Y_train[p]
            # Display logs per epoch step
            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
        print "Optimization Finished!"
        
        # Test model 1
        correct = 0
        #correct_prediction = tf.equal(tf.argmax(tf.reduce_mean(pred, 0, keep_dims=True), 1), tf.argmax(y, 1))
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        Vid_test = data.get_vid_info('Test')
        for i in range(len(Vid_test)):
            batch_x_img = data.get_vid_data(Vid_test[i], 'Img')
            batch_y = np.asarray([int(Vid_test[i].split()[2])])
            batch_y = dense_to_one_hot(batch_y)
     
            vote = np.sum(dense_to_one_hot(tf.argmax(pred, 1).eval({x: batch_x_img, y: batch_y})), axis=0)
            if np.argmax(vote) == np.argmax(batch_y):
                correct += 1
            #batch_y = np.repeat(batch_y, [len(batch_x_img)], axis=0)
            #true = np.sum(correct_prediction.eval({x: batch_x_img, y: batch_y}))
            #if true > len(batch_x_img)/2:
            #    correct+=1        
        # Calculate accuracy
        vid_acc = str(float(correct)/float(len(Vid_test)))
        
        # Test model 2
        X_img_test = data.get_img_X_test()
        y_test = data.get_y_test()
        Y_test = dense_to_one_hot(y_test)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        frame_acc = str(accuracy.eval({x: X_img_test, y: Y_test}))
        
        # Write file
        f = open('DNNIMAGE.txt', 'a')
        f.write(vid_acc + ' ' + frame_acc)
        f.write('\n')
        f.close()