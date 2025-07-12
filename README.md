# 🔬 Bioreactor Model Factory

A full-stack ML pipeline for modeling bioprocess variables (e.g. Dissolved Oxygen, pH) in a fermentation system. This dashboard enables **real-time anomaly detection**, **what-if simulations**, **model interpretability** with SHAP, and **LLM-powered explanations** — all in one interface.

> Built with 🧠 Scikit-learn + 🧪 SHAP + ⚡ Streamlit + 🤖 OpenAI

Demo Video: https://www.loom.com/share/0a0db17bb29047c0ae7c013284048ce1?sid=b4925f00-bd76-4cc5-863b-09b1d098350f
---

## 🌟 Features

| Module                    | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| ✅ What-If Simulator      | Simulate predicted output from user-defined sensor/process inputs           |
| ✅ Real-Time Monitoring   | Detect anomalies using residuals between sensor readings and model output   |
| ✅ SHAP Interpretability  | Understand model decisions using SHAP summary + dependence plots            |
| ✅ LLM Explainer          | Ask GPT-3.5 to explain model behavior and drivers in plain language         |
| ✅ MLflow Model Registry  | Track experiments and manage model versions     
---

## 📊 Dashboard Preview

### 🧪 What-If Simulation
Simulate predictions by entering sensor values:
<img width="874" height="914" alt="Whatif" src="https://github.com/user-attachments/assets/973e7b04-00a6-4bbe-94c4-234181740618" />

---

### 🔴 Real-Time Monitoring + Anomaly Detection
Live residual tracking with anomaly flags:
<img width="707" height="988" alt="real-time anaomly detection" src="https://github.com/user-attachments/assets/d0e01f9a-8576-4cbc-870e-a46058b4ce20" />

---

### 🧬 SHAP + LLM Explainer
Explain model output and features using SHAP + GPT:
<img width="689" height="977" alt="Screenshot 2025-07-12 164053" src="https://github.com/user-attachments/assets/72f0f1c2-5b6a-458f-a569-ab83cbaba991" />

---

## 🧰 Tech Stack

### ⚙️ ML Pipeline

- **Models**:  
  - `RandomForestRegressor` (main)  
  - Optional: `XGBoost`, `LightGBM`, `HistGradientBoosting`, `Ridge`, `SVR`
- **Model Tracking**: `MLflow` (with UI)
- **Training Interface**: Jupyter Notebooks + Python CLI

### 🔎 Interpretability

- **SHAP**:  
  - Summary plots  
  - Dependence plots (top features per target)
- **LLM Explainer**:  
  - OpenAI GPT-3.5 Turbo

### 🖥️ Dashboard

- **Framework**: `Streamlit`
- **Tabs**:
  - `What-If Simulation`
  - `Real-Time Monitoring`
  - `SHAP + Explainer`

### 📈 Data

- **Cleaned Input**: `data/df_clipped_v1.csv`  
- **Simulated Live Feed**: `data/realtime.csv` with noise + anomaly injection

### 💻 Dev Tools

- IDE: `VS Code`  
- Version Control: `Git + GitHub`  
- Env: `.venv + requirements.txt`

---

## 🚀 Run Locally

```bash
# Step 1: Clone repo
git clone https://github.com/Muyangli176/bioreactor-model-factory.git
cd bioreactor-model-factory

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Add your OpenAI key to .streamlit/secrets.toml
# .streamlit/secrets.toml
openai_key = "sk-..."

# Step 4: Launch dashboard
streamlit run dashboard/app.py
