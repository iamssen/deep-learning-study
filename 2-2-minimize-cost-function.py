import tensorflow as _tf
import matplotlib.pyplot as plt

tf = _tf.compat.v1
tf.disable_v2_behavior()

X = [1, 2, 3]
Y = [1, 2, 3]

# CAPTURE REDUCE MEAN VALUES

W = tf.placeholder(tf.float32)

# H(x) = Wx
hypothesis = W * X

# Math.pow(H(x) - y, 2)
cost_fn = tf.square(hypothesis - Y)
cost = tf.reduce_mean(cost_fn)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-30,  50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost,  W], feed_dict={W: feed_W})

    W_val.append(curr_W)
    cost_val.append(curr_cost)

# START TRAIN

W = tf.Variable(tf.random_normal([1]), name='weight')

hypothesis = W * X

cost_fn = tf.square(hypothesis - Y)
cost = tf.reduce_mean(cost_fn)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)

train_W, train_cost = sess.run([W, cost])

# PRINT OUT

plt.plot(W_val, cost_val, label='cost')
plt.plot(train_W, train_cost, marker='.', markersize=20, label='trained cost')
plt.legend()
plt.show()