import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load available models
model_dir = "models"
available_models = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]

st.title("🧪 Sanofi Model Dashboard")

selected_model = st.selectbox("Choose a model:", available_models)

# Load model
model_path = os.path.join(model_dir, selected_model)
model = joblib.load(model_path)

# Load data for prediction
df = pd.read_csv("data/df_clipped_v1.csv")

# Infer target name from file name
y_name = selected_model.split("_")[0]
y = df[y_name]
X = df.drop(columns=[y_name, "SAP", "Age"])

# Make prediction
y_pred = model.predict(X)

# Plot
st.subheader("📈 Prediction vs Actual")
fig, ax = plt.subplots()
sns.scatterplot(x=y, y=y_pred, alpha=0.5, ax=ax)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal')
ax.set_xlabel("True")
ax.set_ylabel("Predicted")
ax.set_title(f"{y_name} Prediction")
st.pyplot(fig)
