# model.py

import tensorflow as tf

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(9, activation='linear')
])
