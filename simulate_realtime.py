import pandas as pd
import numpy as np
import os

# === CONFIG ===
input_path = "data/df_clipped_v1.csv"
output_dir = "data/realtime"
os.makedirs(output_dir, exist_ok=True)

targets = [
    "Dissolved_Oxygen",
    "Fermentor_pH_Probe_B",
    "Fermentor_Skid_Pressure"
]

# === Load base data
df = pd.read_csv(input_path)

for target in targets:
    df_live = df.sample(n=200, random_state=42).reset_index(drop=True)

    # Add timestamp
    df_live["Timestamp"] = pd.date_range("2025-01-01", periods=200, freq="5min")

    # Add noise + anomalies
    noise = np.random.normal(0, 1.0, size=len(df_live))
    df_live[target] += noise
    df_live.loc[df_live.sample(5, random_state=1).index, target] += np.random.normal(10, 2, size=5)

    # Reorder
    cols = ["Timestamp"] + [c for c in df_live.columns if c != "Timestamp"]
    df_live = df_live[cols]

    # Save
    out_file = os.path.join(output_dir, f"{target}_realtime.csv")
    df_live.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file}")
