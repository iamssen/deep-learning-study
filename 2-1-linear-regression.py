import tensorflow as _tf
import matplotlib.pyplot as plt

tf = _tf.compat.v1
tf.disable_v2_behavior()

# Training Set
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# H(x) = Wx + b
# H(x) = 1x + 0
hypothesis = W * x_train + b  # Tensor("add:0", shape=(3,), dtype=float32)

# Math.pow(H(x) - y, 2)
# 1. convert to positive number
# 2. the bigger the difference, the bigger the penalty
cost_fn = tf.square(hypothesis - y_train)
# Tensor("Mean:0", shape=(), dtype=float32)
cost = tf.reduce_mean(cost_fn)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
step_val = []
W_val = []
b_val = []
print('W:', sess.run(W))

for step in range(2001):
    sess.run(train)

    curr_cost, curr_W, curr_b = sess.run([cost, W, b])
    step_val.append(step)
    cost_val.append(curr_cost)
    W_val.append(curr_W)
    b_val.append(curr_b)

    if step % 20 == 0:
        print(
            step,
            'cost:', sess.run(cost),
            'W:', sess.run(W),
            'b:', sess.run(b)
        )

# cost is the difference of H(x[i]) and y[i], it will approach 0
plt.plot(step_val, cost_val, label='cost')
# W and b are variables that settled by tensorflow, those will approach 1 and 0
plt.plot(step_val, W_val, label='W')
plt.plot(step_val, b_val, label='b')
plt.legend()
plt.show()