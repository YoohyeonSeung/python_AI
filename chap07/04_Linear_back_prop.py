import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1.],[2.],[3.]]
y_data = [[1.],[2.],[3.]]

X = tf.placeholder(tf.float32, shape=[None, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.truncated_normal([1,1]))
b = tf.Variable(5.)

# 순전파
hypothesis = tf.matmul(X, W) + b

# diff
assert hypothesis.shape.as_list() == Y.shape.as_list()
diff = (hypothesis - Y)

# 역전파
d_l1 = diff # 예측된 신호와 실제 신호의 차
d_b = d_l1 # d_ : 편미분
d_w = tf.matmul(tf.transpose(X), d_l1)

print(X, W, d_l1, d_w)


learning_rate = 0.1

step = [tf.assign(W, W-learning_rate * d_w), tf.assign(b, b-learning_rate * tf.reduce_mean(d_b))]


rmse = tf.reduce_mean(tf.square(Y-hypothesis))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10001):
        if i%20==0:
            print(i, sess.run([step, rmse], feed_dict={X:x_data, Y:y_data}))

    print(sess.run(hypothesis, feed_dict={X:x_data}))