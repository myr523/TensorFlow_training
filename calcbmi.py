#! /use/bin/python
# coding: utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
# get data
csv = pd.read_csv("bmi.csv")

# normalization
csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 100

# represent the label as 3D class
bclass = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}
csv["label_pat"] = csv["label"].apply(lambda x :np.array(bclass[x]))

# set test data to get correct ans rate
test_csv = csv[15000:20000]
test_pat = test_csv[["weight", "height"]]
test_ans = list(test_csv["label_pat"])

# construct data flow graph
# declare placeholder to input data
x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 3])

# declare var
W = tf.Variable(tf.zeros([2,3]))
b = tf.Variable(tf.zeros([3]))

# define softmax regressoon
y = tf.nn.softmax(tf.matmul(x, W) + b)

# training model
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

# calc correct ans rate
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

# start session
sess = tf.Session()
# init val
sess.run(tf.initialize_all_variables())

# learning
for step in range(3500):
    i = (step * 100) % 14000
    rows = csv[1 + i : 1 + i + 100]
    x_pat = rows[["weight", "height"]]
    y_ans = list(rows["label_pat"])
    fd = {x: x_pat, y_: y_ans}
    sess.run(train, feed_dict=fd)
    if step % 500 == 0:
        cre = sess.run(cross_entropy, feed_dict=fd)
        acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
        print("step=", step, "cre=", cre, "acc=", acc)

acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
print("correct ans rate = ", acc)
