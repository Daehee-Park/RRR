import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
file_path='https://raw.githubusercontent.com/Daehee-Park/Resin/main/Resin_Data.csv'
df = pd.read_csv(file_path)

# Split the data into features and target
X = df[["Resin", "Thickness", "Time", "Weight"]]
y = df["Length"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = SVR(kernel='linear')
model.fit(X_train, y_train)

def predict_length(resin, thickness, time, weight):
    # Make predictions
    y_pred = model.predict(scaler.transform(np.array([resin, thickness, time, weight]).reshape(1, -1)))
    return y_pred[0]

st.title("Length Prediction App")

resin = st.sidebar.selectbox("Select Resin", df["resin"].unique().tolist())
thickness = st.sidebar.slider("Thickness", min_value=df["thickness"].min(), max_value=df["thickness"].max(), value=df["thickness"].mean(), step=0.1)
time = st.sidebar.slider("Time", min_value=df["time"].min(), max_value=df["time"].max(), value=df["time"].mean(), step=0.1)
weight = st.sidebar.slider("Weight", min_value=df["weight"].min(), max_value=df["weight"].max(), value=df["weight"].mean(), step=0.1)

result = predict_length(resin, thickness, time, weight)
st.write("Predicted Length:", result)

# Plot the Time-Length graph
time = np.arange(0, max(df["time"]), 0.1)
length = model.predict(scaler.transform(np.array([0, 0, time, 0]).reshape(len(time), -1)))
plt.plot(time, length, 'g--', label="Predicted Length")
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("Length")
st.pyplot()
