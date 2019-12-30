import tensorflow as _tf

tf = _tf.compat.v1
tf.disable_v2_behavior()

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(5.0, name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# H(x) = Wx
hypothesis = W * X

# Math.pow(H(x) - y, 2)
cost_fn = tf.square(hypothesis - Y)

cost = tf.reduce_mean(cost_fn)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(
            step,
            sess.run(cost, feed_dict={X: x_data, Y: y_data}),
            sess.run(W)
        )

# USING HYPOTHESIS

print('H([5]) is', sess.run(hypothesis, feed_dict={X: [5]}))
print('H([2.5]) is', sess.run(hypothesis, feed_dict={X: [2.5]}))
print('H([1, 2, 3, 4, 5]) is', sess.run(
    hypothesis, feed_dict={X: [1, 2, 3, 4, 5]}))
