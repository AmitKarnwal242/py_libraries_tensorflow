import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print("Model Architecture:")
model.summary()

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True
)

print("Model Diagram Saved Successfully")