import tensorflow as tf

# Create two matrices
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Element-wise multiplication
result = tf.multiply(a, b)

print("Matrix A:")
print(a.numpy())

print("Matrix B:")
print(b.numpy())

print("Multiplication Result:")
print(result.numpy())