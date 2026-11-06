import tensorflow as tf

# Multiple inputs
input1 = tf.keras.Input(shape=(1,), name="input1")
input2 = tf.keras.Input(shape=(1,), name="input2")

# Combine inputs
combined = tf.keras.layers.Concatenate()([input1, input2])

# Hidden layer
hidden = tf.keras.layers.Dense(8, activation='relu')(combined)

# Multiple outputs
output1 = tf.keras.layers.Dense(1, name="output1")(hidden)
output2 = tf.keras.layers.Dense(1, activation='sigmoid', name="output2")(hidden)

# Create model
model = tf.keras.Model(
    inputs=[input1, input2],
    outputs=[output1, output2]
)

model.summary()

print("Functional API Model Created Successfully")