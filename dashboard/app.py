# streamlit run dashboard/app.py
import os
import openai
import streamlit as st
import pandas as pd
from PIL import Image

# === 🔐 Load OpenAI API key from .streamlit/secrets.toml ===
openai.api_key = st.secrets["openai_key"]

# === CONFIG ===
PLOT_DIR = "outputs/plots"
EXPERIMENTS = ["Dissolved_Oxygen", "Fermentor_pH_Probe_B", "Fermentor_Skid_Pressure"]

# === Streamlit Setup ===
st.set_page_config(page_title="Sanofi Model Dashboard", layout="centered")
st.title("🔬 Sanofi Model Factory Dashboard")

# Tab layout
tab1, tab2, tab3 = st.tabs(["🧪 What-If Simulation", "📡 Real-Time Monitoring", "📊 SHAP + Explainer"])

with tab1:
    # === Target Selection ===
    y_target = st.selectbox("📌 Select Target", EXPERIMENTS)

    # === What-If Simulation ===
    import joblib

    with st.expander("🧪 What-If Simulation (Custom Input Prediction)", expanded=True):
        st.markdown("Enter sensor/process values to simulate the predicted output:")

        # Load the trained model
        model_path = f"models/{y_target}_RandomForestRegressor.joblib"
        model = joblib.load(model_path)

        # Load sample to extract feature structure
        df_sample = pd.read_csv("data/df_clipped_v1.csv")
        feature_cols = df_sample.drop(columns=["SAP", "Age", y_target], errors="ignore").columns

        user_input = {}
        for col in feature_cols:
            default = float(df_sample[col].mean())
            user_input[col] = st.number_input(col, value=default, key=f"input_{col}")

        if st.button("🔮 Predict", key="simulate_predict"):
            df_input = pd.DataFrame([user_input])
            pred = model.predict(df_input)[0]
            st.success(f"Predicted **{y_target}**: `{pred:.3f}`")
with tab3:
    # === SHAP Summary ===
    summary_path = next(
        (os.path.join(PLOT_DIR, f) for f in os.listdir(PLOT_DIR) if y_target in f and "summary" in f),
        None
    )

    if summary_path:
        st.subheader("📊 SHAP Summary")
        zoom = st.checkbox("🔍 Full-Size Summary View")
        img = Image.open(summary_path)
        if zoom:
            st.image(img, use_column_width=True)
        else:
            st.image(img.resize((900, 500)), use_column_width=False)
    else:
        st.warning("No SHAP summary plot found for this target.")

    # === SHAP Dependence ===
    dep_plots = [f for f in os.listdir(PLOT_DIR) if y_target in f and "dep" in f]
    if dep_plots:
        st.subheader("🔍 Top Feature Dependence")
        zoom_dep = st.checkbox("🔍 Full-Size Dependence Plots")
        for plot in dep_plots:
            img_path = os.path.join(PLOT_DIR, plot)
            img = Image.open(img_path)
            if zoom_dep:
                st.image(img, caption=plot.replace(".png", ""), use_column_width=True)
            else:
                st.image(img.resize((900, 500)), caption=plot.replace(".png", ""), use_column_width=False)
    else:
        st.info("No dependence plots found for this target.")
    # === Model Explainer ===
    st.subheader("💬 Ask the Model Explainer")
    user_q = st.text_input("What would you like to ask about this model?", placeholder="e.g., Why is airflow important?")

    if user_q:
        with st.spinner("Thinking..."):
            context = f"""
            This is a bioprocess model for predicting the value of **{y_target}**.
            The model is a Random Forest trained on sensor and bioreactor data from fermentation runs.
            SHAP values were used to interpret the feature importance.
            Top SHAP features identified include:
            {', '.join([p.replace('.png', '').split('_dep_')[-1] for p in dep_plots])}.
            """
            prompt = f"{context}\n\nUser question: {user_q}\n\nGive a helpful and technically accurate answer:"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            answer = response.choices[0].message["content"]
            st.success(answer)

            os.makedirs("outputs/logs", exist_ok=True)
            with open("outputs/logs/qna_log.txt", "a") as f:
                f.write(f"[{y_target}] {user_q} -> {answer}\n")

with tab2:
    st.subheader("📡 Real-Time Monitoring (Anomaly Detection)")
    try:
        live_path = f"data/realtime/{y_target}_realtime.csv"
        df_live = pd.read_csv(live_path)

        X_live = df_live.drop(columns=["SAP", "Age", "Timestamp", y_target], errors="ignore")
        y_actual = df_live[y_target]
        y_pred = model.predict(X_live)

        df_live["Predicted"] = y_pred
        df_live["Residual"] = y_actual - y_pred
        df_live["Anomaly"] = df_live["Residual"].abs() > 2.0

        st.dataframe(df_live[["Timestamp", y_target, "Predicted", "Residual", "Anomaly"]].tail(20))
        st.line_chart(df_live.set_index("Timestamp")[[y_target, "Predicted"]].tail(100))

        anomalies = df_live[df_live["Anomaly"]]
        if not anomalies.empty:
            st.warning(f"⚠️ {len(anomalies)} anomalies detected.")

            # Choose anomaly to explain
            idx = st.selectbox("🧠 Select an anomaly row to explain", anomalies.index)
            st.write(f"Row timestamp: {df_live.loc[idx, 'Timestamp']}")
            
            # SHAP explanation
            import shap
            X_row = X_live.iloc[[idx]]
            explainer = shap.Explainer(model.named_steps["model"]) if hasattr(model, "named_steps") else shap.Explainer(model)
            shap_vals = explainer(X_row)

            st.subheader("🔎 Local SHAP Explanation")
            st.pyplot(shap.plots.waterfall(shap_vals[0], max_display=6, show=False))

            # GPT summary
            top_feats = sorted(zip(X_row.columns, shap_vals[0].values), key=lambda x: abs(x[1]), reverse=True)[:3]
            top_features_str = ", ".join(f"{f[0]} ({f[1]:.2f})" for f in top_feats)
            anomaly_prompt = f"""
            The model predicted {df_live.loc[idx, 'Predicted']:.2f} for {y_target}, but the true value was {df_live.loc[idx, y_target]:.2f}.
            Top contributing features were: {top_features_str}.
            Explain the deviation in under 50 words.
            """
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": anomaly_prompt}],
                temperature=0.3,
                max_tokens=60
            )
            st.markdown(f"🧠 **AI Explanation:** {response.choices[0].message['content']}")

        else:
            st.success("✅ No anomalies detected.")

    except Exception as e:
        st.error(f"Live monitoring error: {e}")

