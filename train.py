import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# MLflow Setup
# --------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris_Classification")


# --------------------------------------------------
# Load Iris Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("X shape:", X.shape)
print("y shape:", y.shape)


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# Hyperparameter Experiments
# --------------------------------------------------

experiments = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 300, "max_depth": None}
]


# --------------------------------------------------
# Run Experiments
# --------------------------------------------------

for config in experiments:

    print("\n" + "=" * 50)
    print(
        f"Running experiment: "
        f"n_estimators={config['n_estimators']}, "
        f"max_depth={config['max_depth']}"
    )

    with mlflow.start_run():

        # -------------------------
        # Create Model
        # -------------------------

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42
        )

        # -------------------------
        # Train
        # -------------------------

        model.fit(X_train, y_train)

        # -------------------------
        # Predictions
        # -------------------------

        y_pred = model.predict(X_test)

        # -------------------------
        # Metrics
        # -------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted"
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted"
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted"
        )

        # -------------------------
        # Log Parameters
        # -------------------------

        mlflow.log_params(config)

        mlflow.log_param(
            "random_state",
            42
        )

        # -------------------------
        # Log Metrics
        # -------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # -------------------------
        # Log Model
        # -------------------------

        mlflow.sklearn.log_model(
            model,
            "random_forest_model"
        )

        # -------------------------
        # Print Results
        # -------------------------

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print(
            "MLflow Run ID:",
            mlflow.active_run().info.run_id
        )


print("\nAll experiments completed successfully.")