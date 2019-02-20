import tensorflow as tf

tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name = "Weight")
b = tf.Variable(tf.random_normal([1]), name = "Bias")

X = tf.placeholder(tf.float32, shape = [None])
Y = tf.placeholder(tf.float32, shape = [None])

# 가설 함수 H(x)
hypothesis = X * W + b

# cost 함수
cost = tf.reduce_mean( tf.square(hypothesis - Y) )

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#밑의 for구문만 수정을 해주면 됨. 위의 코드는 수정 할 필요가 없다.
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], \
                                         feed_dict={X:[1, 2, 3], Y:[1, 2, 3]}) # placeholder로 선언한 변수들을 넣어준다.
    # feed_dict를 이용하여 X, Y값을 조정할 수 있음 이전의 py파일과 다른방법
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5, 3.5, 4.5]}))


for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], \
                                         feed_dict={X:[1, 2, 3], Y:[5, 7, 6]})
    # feed_dict를 이용하여 X, Y값을 조정할 수 있음 이전의 py파일과 다른방법
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5, 3.5, 4.5]}))