import tf from '@tensorflow/tfjs-node';

const x_train = tf.tensor([1, 2, 3]);
const y_train = tf.tensor([1, 2, 3]);

const W = tf.variable(tf.randomNormal([1]), undefined, 'weight');
const b = tf.variable(tf.randomNormal([1]), undefined, 'bias');

// const hypothesis = x_train * W + b;
const hypothesis = tf.tidy(() => {
  return x_train.mul(W).sum(b);
});

// tf.minimize