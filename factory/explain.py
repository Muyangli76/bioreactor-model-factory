import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_and_save_shap_summary(model_pipeline, X_raw, feature_names, output_path, title="SHAP_Summary", max_samples=100):
    """
    Saves SHAP summary plot and dependence plots for top 3 features.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = model_pipeline.named_steps["model"]
    if len(model_pipeline.steps) > 1:
        preprocessor = model_pipeline[:-1]
        X_proc = preprocessor.transform(X_raw)
    else:
        X_proc = X_raw.copy()

    X_sample = X_proc[:max_samples] if hasattr(X_proc, '__getitem__') else X_proc.to_numpy()[:max_samples]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 🌈 SHAP Summary Plot
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    summary_path = os.path.join(output_path, f"{title}_summary.png")
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()

    # 💡 Auto-save Top 3 Dependence Plots
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-3:][::-1]
    top_features = [feature_names[i] for i in top_idx]

    for feat in top_features:
        shap.dependence_plot(feat, shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        dep_path = os.path.join(output_path, f"{title}_dep_{feat}.png")
        plt.savefig(dep_path)
        plt.close()
