#! /use/bin/python
import tensorflow as tf

# define constant
a = tf.constant(120, name = "a")
b = tf.constant(130, name = "b")
c = tf.constant(140, name = "c")

# define variable
v = tf.Variable(0, name = "v")

# set formula
calc_op = a + b + c
assign_op = tf.assign(v, calc_op)

# processing
sess = tf.Session()
sess.run(assign_op)

print(sess.run(v))
