import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
# 당뇨병 환자들의 데이터
data = np.loadtxt('data-03-diabetes.csv', delimiter=",", dtype=np.float32)
#1~749개로 training, 750~759까진 test

x_tra_data = data[:-10, 0:-1]
y_tra_data = data[:-10, [-1]]

x_test = data[749:,0:-1]
y_test = data[749:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1] )

W = tf.Variable(tf.random_normal([8, 1]), "Weight")
b = tf.Variable(tf.random_normal([1]), "Bias")

# Hypothesis 함수
Hypothesis = tf.sigmoid(tf.matmul(X, W) +b )

# cost 함수
cost = -tf.reduce_mean(Y*tf.log(Hypothesis)+(1-Y)*tf.log(1-Hypothesis), axis=1)

# 경사하강법
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

predict = tf.cast(Hypothesis > 0.5, dtype=tf.float32)
accurancy = tf.reduce_mean( tf.cast(tf.equal(predict, Y), dtype = tf.float32 ))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000000):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_tra_data, Y:y_tra_data})
        if step%200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([Hypothesis, predict, accurancy], feed_dict={X:x_tra_data, Y:y_tra_data})
    print("\nAccurancy : ",a)


