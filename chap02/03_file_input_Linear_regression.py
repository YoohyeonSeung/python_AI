import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

data = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype = np.float32) # delimiter는 구분자로 어떤걸 할건지 알려주는 명령어

x_data = data[: , 0:-1]  #행은 다 가져오고 열은 0~2까지 가져오는 것
y_data = data[:, [-1]] # 마지막 열만 가지고 오는 것

print(x_data.shape)





# X = tf.placeholder(tf.float32, shape=[None, 3])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
#
# W = tf.Variable(tf.random_normal([3, 1]), name = "weight")
# b = tf.Variable(tf.random_normal([1]), name = "bias")
#
# # Hypothesis 함수
# hypothesis = tf.matmul(X, W) + b
#
# # cost 함수
# cost = tf.reduce_mean(tf.square( hypothesis - Y  ))
#
# # Gradient descent algorithm
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(cost)
#
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(8001):
#     cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
#     if step%100 == 0 :
#         print("step : ", step, "cost : ", cost_val, "\n prediction :", hy_val)
#
# print("Your score will be ", sess.run(hypothesis   , feed_dict={X : [[77, 88, 60]]}  ))