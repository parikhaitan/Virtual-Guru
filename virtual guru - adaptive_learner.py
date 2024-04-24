import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np

# Model parameters
num_features = 5  # Number of input features
num_hidden_layers = 2
hidden_layer_size = 64
learning_rate = 0.001
lambda_feedback = 0.1

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu', input_shape=(num_features,)))
for _ in range(num_hidden_layers - 1):
    model.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu'))
model.add(tf.keras.layers.Dense(1))

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()

# Flask app
app = Flask(__name__)

# Initialize TensorFlow session and model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = np.array(data['features']).reshape(1, num_features)
    predicted_score = model.predict(X)[0][0]
    return jsonify({'predicted_score': predicted_score})

# Feedback and training API endpoint
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    X = np.array(data['features']).reshape(1, num_features)
    y_true = np.array(data['target']).reshape(1, 1)
    feedback_value = np.array(data['feedback']).reshape(1, 1)

    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss_value = loss_fn(y_true, y_pred) + lambda_feedback * tf.reduce_mean(feedback_value * y_pred)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return jsonify({'message': 'Feedback received and model updated'})

if name == 'main':
    app.run(debug=True)