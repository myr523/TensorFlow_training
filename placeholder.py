#! /use/bin/python
import tensorflow as tf

# define placeholder
a = tf.placeholder(tf.int32, [3])

# set formula
b = tf.constant(3)
calc_op = a ** b

# processing
sess = tf.Session()
r1 = sess.run(calc_op, feed_dict={a:[32,64,128]})
print r1
