import model
import compile
import train
import evaluate
import save
import load

# Load  JSON data into a Python DataFrame
df = pd.read_json('land_records.json')

# Prepare the data for training
X = df[['pattadarName']].to_numpy()
y = df[['area', 'landNature', 'classification', 'marketValue', 'landStatus', 'landType', 'transactionStatus', 'ppbNumber', 'ekycstatus']].to_numpy()

# Compile the model
model.compile()

# Train the model
model.train(X, y)

# Evaluate the model
model.evaluate()

# Save the model
model.save()

# Load the model
model.load()

# Make a prediction
input_data = {'pattadarName': 'John Doe'}
prediction = model.predict(input_data)

# Print the prediction
print('The predicted area is:', prediction['area'])

# Calculate the accuracy of the model on a held-out test set
test_data = {'pattadarName': ['John Doe', 'Jane Doe']}
test_labels = {'area': [100, 200]}

loss, accuracy = model.evaluate(test_data, test_labels)

# Print the loss and accuracy of the model on the test set
print('The loss is:', loss)
print('The accuracy is:', accuracy)
