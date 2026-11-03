import tensorflow as tf

x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])

result = tf.nn.relu(x)

print("Original Values:", x.numpy())
print("After ReLU:", result.numpy())