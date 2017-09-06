# this is just the demo file that I could run the tensorboard program
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a,b)

with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs',sess.graph)
	print(sess.run(x))

writer.close()
