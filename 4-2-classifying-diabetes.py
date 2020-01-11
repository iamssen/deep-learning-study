import tensorflow as _tf
import matplotlib.pyplot as plt
import numpy as np

tf = _tf.compat.v1
tf.disable_v2_behavior()

xy = np.loadtxt('data/3-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost_fn = Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis)
cost = -tf.reduce_mean(cost_fn)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for step in range(10001):
    cost_val, _  = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})

    if step % 1000 == 0:
      print(step, cost_val)

  predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

  h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
  # print('Hypothesis:', h)
  # print('Correct (Y):', c)
  print('Accuracy:', a)

  assert(sess.run(predicted, feed_dict={X: [[-0.294118,0.487437,0.180328,-0.292929,0.,0.00149028,-0.53117,-0.0333333]]}) == [[0.]])
  assert(sess.run(predicted, feed_dict={X: [[-0.882353,-0.105528,0.0819672,-0.535354,-0.777778,-0.162444,-0.923997,0]]}) == [[1.]])