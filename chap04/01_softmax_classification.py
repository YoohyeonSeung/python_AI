import tensorflow as tf

tf.set_random_seed(777)

x_data = [ [1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 11, 3, 4],
           [4, 1, 2, 9],
           [2, 2, 3, 1],
           [1, 8, 5, 6],
           [9, 2, 3, 1],
           [3, 5, 7, 1] ]    # 8 by 4 data
y_data = [ [0, 0, 1],
           [0, 1, 0],
           [0, 0, 1],
           [0, 1, 0],
           [1, 0, 0],
           [0, 0, 1],
           [0, 0, 1],
           [0, 0, 1]]             # 8 by 3 result

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])


W= tf.Variable(tf.random_normal([4, 3]), name="weight")
b =tf.Variable(tf.random_normal([3]), name = "bias")

# Hypothesis 함수

hypothesis = tf.nn.softmax(    tf.matmul(X, W) +b       )

# cost(loss)
cost =  -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis), axis=1)

# Gradient descent algorithm
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   # 변수 초기화 우선!

    for step in range(2001):
        sess.run(train, feed_dict={X:x_data,  Y:y_data})

        if step %200 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data,  Y:y_data}))


    all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7, 9], [2, 2, 3, 4], [5, 1, 8, 11]]})

    print(all, sess.run(tf.argmax(all, 1)))






