import tensorflow as _tf
import matplotlib.pyplot as plt

tf = _tf.compat.v1
tf.disable_v2_behavior()

x_data = [
    [1, 2],
    [2, 3],
    [3, 1],
    [4, 3],
    [5, 3],
    [6, 2]
]
y_data = [
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
]

X = tf.placeholder(tf.float32, shape=[None, 2]) # None is n - don't know how many
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# H(X) = 1 / 1 + e ** -WtX
# sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
# sigmoid: t => 1 / (1 + Math.exp(-t))
# z = WX
# H(X) = sigmoid(WX)
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# if Y = 1 then (1 - Y) = 0
# if Y = 0 then (1 - Y) = 1
# ylog(H(x)) + (1 - y)log(1 - H(x))
cost_fn = Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis)
cost = -tf.reduce_mean(cost_fn)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for step in range(10001):
    cost_val, _  = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})

    if step % 1000 == 0:
      print(step, cost_val)

  # 예측된 값
  predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
  # 정확도
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

  h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
  print('Hypothesis:', h)
  print('Correct (Y):', c)
  print('Accuracy:', a)

  print(sess.run(hypothesis, feed_dict={X: [[1, 2]]}))