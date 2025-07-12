import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

def train_and_log_model(
    X,
    y,
    y_name,
    model,
    data_version="v1_clipped_clean",
    model_dir="models/",
    scale=True,
    impute=True,
    register=True  # ✅ NEW toggle
):
    X = X.copy()
    y = y.copy()

    valid_y = y.notnull()
    X = X[valid_y]
    y = y[valid_y]

    if not isinstance(model, HistGradientBoostingRegressor):
        valid_X = X.notnull().all(axis=1)
        X = X[valid_X]
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    steps = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    pipeline = Pipeline(steps)

    mlflow.set_experiment("sanofi_model_factory")

    with mlflow.start_run(run_name=f"{y_name}_{type(model).__name__}"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Log params and metrics
        mlflow.log_param("target", y_name)
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("impute", impute)
        mlflow.log_param("scale", scale)
        mlflow.set_tag("data_version", data_version)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Save model locally
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{y_name}_{type(model).__name__}.joblib")
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        # ✅ Register to Model Registry
        if register:
            model_registry_name = f"{y_name}_regressor"
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=model_registry_name
            )
            print(f"📦 Registered: {model_registry_name}")

        return {
            "model": pipeline,
            "r2": r2,
            "mae": mae,
            "y_test": y_test,
            "y_pred": y_pred
        }
