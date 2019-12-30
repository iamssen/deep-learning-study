import tensorflow as _tf
import matplotlib.pyplot as plt
import numpy as np

tf = _tf.compat.v1
tf.disable_v2_behavior()

xy = np.loadtxt('./data/1-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# [row, column] 
# [:, 0:-1] → [all rows, 0 ~ all columns - 1]

# (6, 3) [x1, x2, x3][]
print(x_data.shape, x_data)
# (6, 1) [y][]
print(y_data.shape, y_data)


# TENSORFLOW...

# X = [n, 3]
X = tf.placeholder(tf.float32, shape=[None, 3])

# Y = [n, 1]
Y = tf.placeholder(tf.float32, shape=[None, 1])

# W = [3, 1]
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# H(X) = XW + b
hypothesis = tf.matmul(X, W) + b

# Math.pow(H(X) - Y, 2)
cost_fn = tf.square(hypothesis - Y)

cost = tf.reduce_mean(cost_fn)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(4001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={X: x_data, Y: y_data}
    )

    if step % 50 == 0:
        print(step, 'Cost:', cost_val, 'Prediction:', hy_val)

print(sess.run(hypothesis, feed_dict={X: [[73., 80., 75.]]}))
print(sess.run(hypothesis, feed_dict={X: [[93., 88., 93.]]}))
print(sess.run(hypothesis, feed_dict={X: [[89., 91., 90.]]}))
