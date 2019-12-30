import tensorflow as _tf
import matplotlib.pyplot as plt

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

# Gradient descent algorithm?
# Gradient 경사 descent 내려감
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# PRINT OUT

cost_val = []
step_val = []
W_val = []

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})

    curr_cost, curr_W = sess.run([cost, W], feed_dict={X: x_data, Y: y_data})

    step_val.append(step)
    cost_val.append(curr_cost)
    W_val.append(curr_W)

    print(step, curr_cost, curr_W)

plt.plot(step_val, cost_val, label = 'cost')
plt.plot(step_val, W_val, label = 'W')
plt.legend()
plt.show()

# USING HYPOTHESIS

print('H([5]) is', sess.run(hypothesis, feed_dict={X: [5]}))
print('H([2.5]) is', sess.run(hypothesis, feed_dict={X: [2.5]}))
print('H([1, 2, 3, 4, 5]) is', sess.run(hypothesis, feed_dict={X: [1, 2, 3, 4, 5]}))