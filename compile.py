
# compile.py

import tensorflow as tf
import tensorflow.keras.models as model


# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
