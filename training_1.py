#! /use/bin/python
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(4)

calc = a * b * c

sess = tf.Session()
res = sess.run(calc)
print res
