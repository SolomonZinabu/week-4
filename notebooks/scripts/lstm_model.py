# lstm_model.py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
def preprocess_for_lstm(train_fe):
    # Extract features from the 'Date' column
    if 'Date' in train_fe.columns:
        # Convert the 'Date' column to datetime if it's not already
        train_fe['Date'] = pd.to_datetime(train_fe['Date'])
        
        # Extract useful features from the date (Year, Month, Day, Day of the Week)
        train_fe['Year'] = train_fe['Date'].dt.year
        train_fe['Month'] = train_fe['Date'].dt.month
        train_fe['Day'] = train_fe['Date'].dt.day
        train_fe['DayOfWeek'] = train_fe['Date'].dt.dayofweek
        
        # Drop the original 'Date' column since we now have numeric features
        train_fe = train_fe.drop(columns=['Date'])

    # Drop any other non-numeric columns that aren't useful for modeling
    non_numeric_columns = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
    train_fe = train_fe.drop(columns=non_numeric_columns, errors='ignore')

    # Define features (X) and target (y)
    X = train_fe.drop(columns=['Sales'])  # Features
    y = train_fe['Sales']  # Target

    # Ensure all data is float32
    X = X.astype('float32')
    y = y.astype('float32')

    # Reshape X to 3D array (samples, timesteps, features) for LSTM
    X = np.expand_dims(X.values, axis=2)  # Add the 3rd dimension (timesteps=1)

    # Split the data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test

def train_lstm_model(X_train, y_train, X_test, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)
    return model
