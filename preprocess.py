import numpy as np
import joblib

# Load scaler
def load_scaler():
    return joblib.load("model/scaler.save")

# 🔥 AUGMENTATION FUNCTION
def augment_data(X):
    noise = np.random.normal(0, 0.01, X.shape)
    return X + noise
