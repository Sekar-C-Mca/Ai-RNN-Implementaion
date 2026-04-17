import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import h5py

BASE_DIR = Path(__file__).resolve().parent

# Load model weights and run inference without TensorFlow runtime.
def load_dense_weights(model_path: Path):
    with h5py.File(model_path, 'r') as f:
        w1 = f['model_weights/dense/sequential/dense/kernel'][()]
        b1 = f['model_weights/dense/sequential/dense/bias'][()]
        w2 = f['model_weights/dense_1/sequential/dense_1/kernel'][()]
        b2 = f['model_weights/dense_1/sequential/dense_1/bias'][()]
        w3 = f['model_weights/dense_2/sequential/dense_2/kernel'][()]
        b3 = f['model_weights/dense_2/sequential/dense_2/bias'][()]
    return (w1, b1, w2, b2, w3, b3)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_churn_proba(x: np.ndarray, weights) -> float:
    w1, b1, w2, b2, w3, b3 = weights
    h1 = relu(x @ w1 + b1)
    h2 = relu(h1 @ w2 + b2)
    out = sigmoid(h2 @ w3 + b3)
    return float(out[0, 0])


weights = load_dense_weights(BASE_DIR / 'model.h5')

# Load encoders and scaler (UPDATED NAMES)
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_geo.pkl', 'rb') as file:
    one_hot_geo = pickle.load(file)

scaler_path = BASE_DIR / 'scaler.pkl'
if not scaler_path.exists():
    scaler_path = BASE_DIR / 'scalar.pkl'

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', one_hot_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode Geography (UPDATED VARIABLE)
geo_encoded = one_hot_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_geo.get_feature_names_out(['Geography'])
)

# Combine features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale (UPDATED VARIABLE)
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction_proba = predict_churn_proba(input_data_scaled.astype(np.float32), weights)

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')