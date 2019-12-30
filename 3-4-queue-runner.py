import tensorflow as _tf
import matplotlib.pyplot as plt

tf = _tf.compat.v1
tf.disable_v2_behavior()

filename_queue = tf.train.string_input_producer(
    ['./data/1-test-score.csv'],
    shuffle=False,
    name='filename_queue'
)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch(
    [xy[0:-1], xy[-1:]],
    batch_size=10
)

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

# # BATCH...

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(4001):
    x_data, y_data = sess.run([train_x_batch, train_y_batch])

    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train],
        feed_dict={X: x_data, Y: y_data}
    )

    if step % 100 == 0:
        print(step, 'Cost:', cost_val, 'Prediction:', hy_val)

print(sess.run(hypothesis, feed_dict={X: [[73., 80., 75.]]}))
print(sess.run(hypothesis, feed_dict={X: [[93., 88., 93.]]}))
print(sess.run(hypothesis, feed_dict={X: [[89., 91., 90.]]}))
