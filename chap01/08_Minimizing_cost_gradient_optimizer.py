import tensorflow as tf

tf.set_random_seed(777)

x_data = [1, 2, 3]
y_data = [1, 2, 3]

w = tf.Variable(-3.0)

# 가술 함수
hypothesis = x_data * w  # 계산의 편의성을 위해 bias 생략

# cost 함수

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize : Gradient Descent(미분 : 기울기)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):    # learning rate=0.1로 하였기 때문에 금방 끝남
    print(step, sess.run(w))
    sess.run(train)