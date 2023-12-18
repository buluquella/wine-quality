import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Set the file path for the dataset
current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, '..', '..', 'data', 'winequality-white.csv')

# Load the dataset
data = pd.read_csv(file_path, delimiter=';')

# Select features and target
selected_features = ['volatile acidity', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']
X = data[selected_features]
y = data['quality']

# Convert target to categorical (one-hot encoding)
y = to_categorical(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural network model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred_classes = tf.argmax(y_pred, axis=1)
y_test_classes = tf.argmax(y_test, axis=1)
print("Neural Network Model Results:")
print(classification_report(y_test_classes, y_pred_classes))
print("Accuracy Score:", accuracy_score(y_test_classes, y_pred_classes))
