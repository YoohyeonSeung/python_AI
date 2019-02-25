import tensorflow as tf


tf.set_random_seed(777)

x_data= [[1, 2, 1],
         [1, 3, 2],
         [1, 3, 4],
         [1, 5, 5],
         [1, 7, 5],
         [1, 2, 5],
         [1, 6, 6],
         [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


# Evaluation our model using this thest dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]




X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])


W = tf.Variable(tf.random_normal([ 3,3 ]), name="weight" )
b = tf.Variable(tf.random_normal([ 3  ]), name="bias" )

# 가설 함수, cost 함수, Gradient descent algorithm(이부분은 대부분 복붙해도 된다.)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(     -tf.reduce_sum(Y*tf.log(hypothesis), axis=1)       )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)         # learning rate를 5로 하였을 때 nan이 나오는 이유는 08-learning-rate, evauation.pdf의 p.4 에 있다.
train = optimizer.minimize(cost)

"""
    learning rate 와 traing횟수에 따라 cost의 값이 달라진다.
    그러므로 우리가 횟수와 rate를 수정하면서 cost의 값을 줄이도록 해줘야 한다.
"""



prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accurancy = tf.reduce_mean(tf.cast(is_correct, tf.float32))     # True 이면 1.0, False이면 0.0으로 반환


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())    # 변수의 값 초기화

    for step in range(5001):
        cost_val, W_val, _ = sess.run([cost, W, train], feed_dict={X:x_data, Y:y_data})
        print(step, ",", cost_val, ",", W_val)







































