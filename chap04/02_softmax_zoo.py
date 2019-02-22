import numpy as np
import tensorflow as tf

tf.set_random_seed(777)


data = np.loadtxt("data-04-zoo.csv", delimiter=",", dtype=np.float32)

x_train_data = data[:-5 , :-1]
y_train_data = data[:-5 , [-1]]

x_test_data = data[-5: , :-1 ]
y_test_data = data[-5: , [-1] ]

X = tf.placeholder(tf.float32, shape = [None, 16])
Y = tf.placeholder(tf.int32, shape = [None, 1] )      # one-hot 을 할꺼면 float32가 아니라 int로 해줘야됨

# One-hot 인코딩
# 출력값이 두개중 하나가 아니라 3개 이상의 것중 하나일때 다음의 두 코드가 필수로 필요함!
Y_one_hot = tf.one_hot(Y, 7)   #열의 형태
Y_one_hot = tf.reshape(Y_one_hot, [-1, 7])    #행의 형대로 변경

W = tf.Variable(tf.random_normal([16, 7]), name="weight")    # one_hot에 의하여 1by7 벡터로 출력
b = tf.Variable(tf.random_normal([7]), name="bias")


# Hypothesis function(3분류 이상이기 때문에 softmax, 만약 두 분류면 sigmoid)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cost function
cost = tf.reduce_mean(  tf.nn.softmax_cross_entropy_with_logits(    logits=tf.matmul(X, W)+b, labels= Y_one_hot ) )

# Gradient descent algorithm
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

prediction = tf.argmax(hypothesis, 1)  # 출력중 가장 큰 것!
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accurancy = tf.reduce_mean(  tf.cast( correct_prediction, tf.float32) )


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        sess.run(train, feed_dict={X:x_train_data, Y:y_train_data})

        if step % 200 ==0 :
            c, a = sess.run([cost, accurancy],feed_dict={X:x_train_data, Y:y_train_data})
            print(step, "cost :{:.2f}  Accurancey : {:.2%}".format(c, a))

    pred = sess.run(prediction, feed_dict={X:x_test_data})

    for p, y in zip(pred):
        print( "[{}] Prediction : {} True Y :{}".format(p==int(y), p, int(y))               )