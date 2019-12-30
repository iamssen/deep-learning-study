import tensorflow as _tf
import matplotlib.pyplot as plt

tf = _tf.compat.v1
tf.disable_v2_behavior()

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

# H(x1, x2, x3) = w1x1 + w2x2 + w3x3 + b
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# Math.pow(H(x1, x2, x3) - Y, 2)
cost_fn = tf.square(hypothesis - Y)
cost = tf.reduce_mean(cost_fn)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(4001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={x1: x1_data, x2: x2_data,  x3: x3_data, Y: y_data}
    )

    if step % 50 == 0:
        print(step, 'Cost:', cost_val, 'Prediction:', hy_val)

print(sess.run(hypothesis, feed_dict={x1: [73.], x2: [80.], x3: [75.]}))
print(sess.run(hypothesis, feed_dict={x1: [93.], x2: [88.], x3: [93.]}))
print(sess.run(hypothesis, feed_dict={x1: [89.], x2: [91.], x3: [90.]}))
