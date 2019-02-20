import tensorflow as tf
## pdf 04_cost최소화 알고리즘 p23
tf.set_random_seed(777)

x_date = [1, 2, 3]
y_data = [2, 4, 6]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis-Y))

learning_rate = 0.1
gradient = tf.reduce_mean(   (  W* X -Y )   * X  )
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1001):
    sess.run(update, feed_dict={X:x_date, Y:y_data})
    print(step, sess.run(cost, feed_dict= {X: x_date, Y:y_data}  ),  sess.run(W)   )