import tensorflow as _tf

tf = _tf.compat.v1
tf.disable_v2_behavior()

# Hello World
hello = tf.constant('Hello, Tensorflow!')
sess = tf.Session()

assert(sess.run(hello).decode(encoding='utf-8') == 'Hello, Tensorflow!')

# Computational Graph
a = tf.constant(3., tf.float32)
b = tf.constant(4.)
c = tf.add(a, b)

sess = tf.Session()
assert(sess.run([a, b]) == [3., 4.])
assert(sess.run(c) == [7.])

# Placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = a + b

sess = tf.Session()
assert(sess.run(c, feed_dict={a: 3, b: 4.5}) == 7.5)
assert((sess.run(c, feed_dict={a: [1, 3], b: [2, 4]}) == [3., 7.]).all())

# reduce_mean
n = [1., 2., 3., 4.]
sess = tf.Session()
assert(sess.run(tf.reduce_mean(n)) == 2.5)

# Variable Assign
var = tf.Variable(5., name='variable')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

assert(sess.run(var) == 5.)

sess.run(var.assign(4.))
assert(sess.run(var) == 4.)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = a + b

sess.run(var.assign(c), feed_dict={a: 5, b: 8})
assert(sess.run(var) == 13.)
