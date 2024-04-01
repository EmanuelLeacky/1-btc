import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from joblib import load

# Define look_back globally if its value is constant throughout the script
look_back = 1  # Make sure this is the value you intend to use

# Disable the use of Matplotlib's backend
plt.switch_backend('agg')

# Ensure TensorFlow does not allocate all memory when loading the model
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        st.text(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        st.error(f"Failed to set memory growth for TensorFlow: {e}")

# Function to preprocess data for the LSTM model
def create_dataset(dataset, look_back):
    X = []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
    return np.array(X)

# Function to predict prices using the model and scaler
def predict_prices(model, scaler, data, look_back):
    # Reshape the data to [-1, 1] since we only have one feature
    scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))
    X_test = create_dataset(scaled_data, look_back)
    # Reshape input to [samples, time steps, features] for LSTM
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predictions = model.predict(X_test)
    # Inverse transform to get actual prices
    return scaler.inverse_transform(predictions)

st.title('Bitcoin Price Prediction')

# Attempt to load the trained model and scaler with error handling
try:
    # Load the model with custom objects if necessary
    model = tf.keras.models.load_model('btc_usd_trained_model.keras', compile=False)
    scaler = load('btc_scaler.bin')
    st.success("Model and scaler loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Sidebar for file uploading
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Read and preprocess uploaded data
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])

    # Display raw data if checkbox is selected
    if st.checkbox('Show raw data'):
        st.write(data)

    # Plotting actual prices
    st.subheader('Bitcoin Closing Price Over Time')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['Date'], data['Close'], label='Actual Prices', color='blue', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Button to predict prices
    if st.button('Predict Bitcoin Prices'):
        with st.spinner('Predicting Bitcoin Prices...'):
            predictions = predict_prices(model, scaler, data, look_back)
        
        # Adjust dates for plotting predictions, accounting for look_back
        pred_dates = data['Date'][look_back:].reset_index(drop=True)

        # Plotting predictions
        st.subheader('Model Predictions vs Actual Prices')
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(pred_dates, predictions.flatten(), label='Predictions', color='red', alpha=0.6)  # Ensure predictions are flattened for plotting
        ax2.plot(data['Date'], data['Close'], label='Actual Prices', color='blue', alpha=0.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Create a DataFrame with the predicted prices and dates
        pred_df = pd.DataFrame({'Date': pred_dates, 'Predicted Price': predictions.flatten()})
        st.subheader('Predicted Bitcoin Prices Table')
        st.table(pred_df)
else:
    st.warning("Please upload a CSV file to proceed.")
