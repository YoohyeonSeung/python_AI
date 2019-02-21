import tensorflow as tf

tf.set_random_seed(777)

# matrix(시험을 치룬 사람을 한 행으로 묶어서 하는 경우)
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.], 
          [185.], 
          [180.], 
          [196.], 
          [142.]]

X = tf.placeholder(tf.float32, shape=[None, 3]) #입력하는 한 단위의 형태, 몇개가 들어올지 모르니 None
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name = "weight")  # 입력으로 들어오는 한단위의 열의 갯수와 동일
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis 함수
hypothesis = tf.matmul(X, W) + b

# cost 함수
cost = tf.reduce_mean(tf.square( hypothesis - Y  ))

# Gradient descent algorithm
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(8001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
    if step%100 == 0 :
        print("step : ", step, "cost : ", cost_val, "\n prediction :", hy_val)

print("Your score will be ", sess.run(hypothesis   , feed_dict={X : [[77, 88, 60]]}  ))