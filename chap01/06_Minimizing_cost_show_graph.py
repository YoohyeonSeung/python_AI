import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)

X = [1, 2, 3]
Y = [2, 4, 6]

W = tf.placeholder(tf.float32)

# 가설 함수
hypothesis = X * W

# cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()

w_history = []
cost_history = []

for i in range(-30, 80):
    curr_W = i*0.1
    curr_cost = sess.run(cost, feed_dict={W:[curr_W]})
    w_history.append(curr_W)
    cost_history.append(curr_cost)


plt.plot(w_history, cost_history)
plt.show()