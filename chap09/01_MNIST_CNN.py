import tensorflow as tf
import random
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   # MNIST_data폴더를 만들어서 파일 저장


# Hyper-parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 100


X = tf.placeholder(tf.float32,  [None, 784]  )
X_img = tf.reshape(X, [-1, 28, 28, 1 ])
Y = tf.placeholder(tf.float32,  [None, 10]  )


W1 = tf.Variable(tf.random_normal([ 3, 3, 1, 32], stddev=0.01))   # 높이 너비3 , 1개채널이 32개
#       conv        ->    (?, 28, 28, 32)
#       pooling     ->    (?, 14, 14, 32)
L1 = tf.nn.conv2d( X_img, W1, strides=[1, 1, 1, 1]  , padding='SAME' )  # stride는 한칸씩 이동하면서 처리 해라
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')     # 풀링의 윈도우크기와 스트라이드는 같은 값으로 설정하는 것이 일반적.



# L2 ImgIn shape = (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))  # n n n 64 (출력의 갯수 임의 지정)
#     conv          -> (?, 14, 14, 64)
#     pooling       -> (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7*7*64])

# Final FC 7*7*64 inputs -> 10 outputs   # (Fully? Connection)
W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))



hypothesis = tf.matmul(L2_flat, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,  labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):  # 6만개 데이터를 15번 반복
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)  # 6만개의 데이터 / 100 = 600개만 가져옴

        for i in range(total_batch):  # 100개단위의 데이터를 600번 반복
            batch_xs, batch_ys = mnist.train.next_batch(100)
            feed_dict = {X: batch_xs, Y: batch_ys}

            c, _ = sess.run([cost, train], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch: ', '%04d' % (epoch + 1), 'cost : ', '{:.9f}'.format(avg_cost))

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accurnacy : ', sess.run(accurancy, feed_dict={X: mnist.test.images, Y: mnist.test.labels,}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Lable : ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction : ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

