# Live App

https://ko9jzbjajqvgv4c4mjwvkn.streamlit.app/

# AI ANN Churn Prediction

This project predicts customer churn using a trained Artificial Neural Network (ANN) and a Streamlit web interface.

## Features

- Customer churn probability prediction
- Interactive Streamlit UI for manual input
- Preprocessing with saved encoders and scaler
- Inference from saved model weights (`model.h5`) without requiring TensorFlow at runtime

## Project Structure

- `app.py`: Streamlit application
- `model.h5`: Trained ANN model weights
- `label_encoder_gender.pkl`: Label encoder for gender
- `one_hot_geo.pkl`: One-hot encoder for geography
- `scaler.pkl` / `scalar.pkl`: Feature scaler
- `requirements.txt`: Python dependencies for deployment
- `Revison.ipynb`, `ANN.ipynb`, `experiments.ipynb`, `one.ipynb`: notebooks for experiments/training

## Run Locally

1. Create and activate a virtual environment
2. Install dependencies
3. Start Streamlit

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Input Features

The app uses the following features:

- Credit Score
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Geography (one-hot encoded)

## Output

- Churn probability score
- Classification:
  - Likely to churn (probability > 0.5)
  - Not likely to churn (probability <= 0.5)

## Deployment

This app is deployed on Streamlit Cloud:

https://ko9jzbjajqvgv4c4mjwvkn.streamlit.app/
