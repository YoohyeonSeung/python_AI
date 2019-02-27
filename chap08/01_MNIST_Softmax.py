import tensorflow as tf
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777)


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)   # MNIST_data 폴더를 생성하여 데이터 저장

X = tf.placeholder(tf.float32, [None, 784]) # 평탄화 하면 28 * 28 = 784
Y = tf.placeholder(tf.float32, [None, 10])  # 1~10까지 중 하나로


W = tf.Variable(tf.random_normal([784, 10]), name= "Weight")
b = tf.Variable(tf.random_normal([10]), name="bias")

#가설함수
hypothesis = tf.nn.softmax(  tf.matmul(X, W) + b     )

#손실함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=hypothesis, labels=Y         )  )

#Minimize
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# Test model
is_correct = tf.equal(    tf.arg_max(hypothesis, 1)   , tf.arg_max(Y, 1))   #예측 값과 실제 값 비교
accurancy =  tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 100개 단위 data
            cost_val, _ =  sess.run([cost, train], feed_dict={X: batch_xs, Y: batch_ys})

            avg_cost += cost_val/total_batch

        print("Epoch : ", "%04d" % (epoch+1) , "cost = ", "{:.9}".format(avg_cost))

    print("ML finished")

    print("Accurancy: ", accurancy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1 ) # 숫자 선택
    print("lable : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("prediction : ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]} ))

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')

    plt.show()













