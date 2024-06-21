
# evaluate.py

import tensorflow as tf

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Print the test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)
