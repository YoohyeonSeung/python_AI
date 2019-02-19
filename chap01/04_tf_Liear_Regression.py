# tensorflow를 이용한 Linear Regression

import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)


# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]
# Hypothesis Func. y = Wx + b    w와 b를 랜덤하게 꺼내오는것
W = tf.Variable(tf.random_normal([1]), name = 'Weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 가설 함수( H(x) ) = xW + b
hypothesis = x_train * W + b

# Cost(loss) Func.
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)   # 최초의 학습률은 0.1로 할것
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):   #2000번 기계학습을 해라
    sess.run(train)
    if step % 20  ==0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

