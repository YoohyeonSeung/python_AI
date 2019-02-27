import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

learning_rate = 0.1

x_data = [[0, 0],[0, 1],[1, 0],[1, 1]]
y_data = [ [0],    [1],   [1],   [0] ]  # XOR Gate
# -> And 와 Or 게이트도 해볼것

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)


X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")


hypothesis = tf.sigmoid( tf.matmul(X, W) + b )

cost = -tf.reduce_mean(  Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis)        )

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)


prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accurancy = tf.reduce_mean(tf.cast( tf.equal(prediction, Y), dtype=tf.float32   ))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step%100==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

    h, p, a = sess.run([hypothesis, prediction, accurancy], feed_dict={X: x_data, Y:y_data})
    print("\nHypothesis : ", h, "\nPrediction : ", p, "\nAccurancy : ", a)



"""
내가 만든 코드이고 이 위에 있는 코드는 수업중 코드
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})

        if step%10==0:
            print(step, cost_val)

    h, p, a = sess.run([hypothesis, prediction, accurancy], feed_dict={X:x_data})
    print("\nHypothesis : ", h, "\nPrediction : ", p, "\nAccurancy : ",a)
"""




