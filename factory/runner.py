import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from factory.train_model import train_and_log_model
from factory.explain import generate_and_save_shap_summary  # ✅ SHAP logic

# Load cleaned dataset
df = pd.read_csv("data/df_clipped_v1.csv")

# List of target variables you want to train models for
y_targets = [
    "Dissolved_Oxygen",
    "Fermentor_pH_Probe_B",
    "Fermentor_Skid_Pressure"
]

# Columns to exclude from features
exclude_cols = ["SAP", "Age"]

# Optional: Generate a version tag using timestamp
version_tag = datetime.datetime.now().strftime("v%Y%m%d_%H%M")

# Initialize summary
summary = []

# Loop through targets
for y_target in y_targets:
    print(f"\n▶ Training model for: \033[1m{y_target}\033[0m")

    y = df[y_target]
    X = df.drop(columns=[y_target] + exclude_cols)

    result = train_and_log_model(
        X=X,
        y=y,
        y_name=y_target,
        model=RandomForestRegressor(n_estimators=100),
        model_dir="models/",
        data_version=version_tag
    )

    # ✅ Save SHAP summary
    generate_and_save_shap_summary(
        model_pipeline=result["model"],
        X_raw=X,
        feature_names=X.columns,
        output_path="outputs/plots/",
        title=f"{y_target}_{version_tag}"
    )

    summary.append({
        "Target": y_target,
        "R²": round(result["r2"], 3),
        "MAE": round(result["mae"], 3)
    })

# Show summary
summary_df = pd.DataFrame(summary).sort_values("R²", ascending=False)
print("\n✅ \033[1mModel Factory Summary:\033[0m")
print(summary_df.to_string(index=False))
