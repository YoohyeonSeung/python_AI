import tensorflow as tf
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # MNIST_data 폴더를 생성하여 데이터 저장

# parameter
learning_rate = 0.001
training_epochs = 15
batch_size = 100

## dropout(keep_prob) rate 0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, [None, 784])  # 평탄화 하면 28 * 28 = 784
Y = tf.placeholder(tf.float32, [None, 10])  # 1~10까지 중 하나로

# layer1 구성(active function = ReLU , Xavier), 출력 갯수 : 256
W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
#dropout 적용
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)



W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
#dropout 적용
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# train = optimizer.minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):  # 6만개 데이터를 15번 반복
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)  # 6만개의 데이터 / 100 = 600개만 가져옴

        for i in range(total_batch):  # 100개단위의 데이터를 600번 반복
            batch_xs, batch_ys = mnist.train.next_batch(100)
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob:0.7}  #여기서도 dropout 적용

            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch: ', '%04d' % (epoch + 1), 'cost : ', '{:.9f}'.format(avg_cost))

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accurnacy : ', sess.run(accurancy, feed_dict={X: mnist.test.images, Y:mnist.test.labels, keep_prob:1})) #여기서도 dropout 적용(test이면 1)

    r = random.randint(0, mnist.test.num_examples - 1)  # 데이터셋에서 하나 가져와서 맞추는지 보겟다는것(랜덤이기 때문에 할때마다 다를 수 있음)
    print("Lable : ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction : ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob:1})) #여기서도 dropout 적용(test이면 1)
