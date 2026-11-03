import tensorflow as tf

# Create variable
x = tf.Variable(5.0)

# Calculate gradient
with tf.GradientTape() as tape:
    y = x ** 2

gradient = tape.gradient(y, x)

print("Value of x:", x.numpy())
print("Value of y:", y.numpy())
print("Gradient:", gradient.numpy())