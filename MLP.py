# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Function to load dataset and handle potential errors
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

# Load dataset
file_path = 'weather_data.csv'  # The name of your dataset
data = load_data(file_path)

# Print the columns of the DataFrame
print("Columns in the dataset:", data.columns)

# Clean column names
data.columns = data.columns.str.strip()

# Check if required columns exist in the data
required_columns = ['humidity', 'pressure_mb', 'wind_mph', 'temperature_celsius']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"Missing columns in the data. Required: {required_columns}")

# Select features and target variable
X = data[['humidity', 'pressure_mb', 'wind_mph']].values  # Input features
y = data['temperature_celsius'].values  # Target variable (temperature in Celsius)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data to range between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
    Dropout(0.2),
    Dense(64, activation='relu'),  # Hidden layer
    Dropout(0.2),
    Dense(1)  # Output layer for temperature prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.2, 
                    callbacks=[early_stopping],
                    verbose=1)

# Evaluate on the test set
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.2f}")

# Make predictions
y_pred = model.predict(X_test)

# Plot Actual vs Predicted temperature values
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Temperature', color='blue')
plt.plot(y_pred, label='Predicted Temperature', color='red')
plt.title('Actual vs Predicted Temperature')
plt.xlabel('Sample Index')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Plot training history (loss over epochs)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save('weather_forecasting_model.h5')
print("Model saved as 'weather_forecasting_model.h5'")

loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.2f}")
