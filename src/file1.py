import optuna
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# Load Dataset
# ----------------------------------------
wine = load_wine()

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)

# ----------------------------------------
# MLflow Experiment
# ----------------------------------------
mlflow.set_experiment("YT-MLOPS-Optuna")


# ----------------------------------------
# Optuna Objective Function
# ----------------------------------------
def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None]
        ),
        "criterion": trial.suggest_categorical(
            "criterion",
            ["gini", "entropy", "log_loss"]
        ),
        "bootstrap": trial.suggest_categorical(
            "bootstrap",
            [True, False]
        ),
        "random_state": 42,
        "n_jobs": -1,
    }

    # Nested MLflow run
    with mlflow.start_run(
        nested=True,
        run_name=f"Trial-{trial.number}"
    ):

        model = RandomForestClassifier(**params)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.set_tag("trial_number", trial.number)

        return accuracy


# ----------------------------------------
# Parent Run (Optuna)
# ----------------------------------------
with mlflow.start_run(run_name="Optuna Optimization"):

    study = optuna.create_study(direction="maximize")

    study.optimize(
        objective,
        n_trials=30
    )

    mlflow.log_param("n_trials", 30)
    mlflow.log_metric("best_accuracy", study.best_value)

    for key, value in study.best_params.items():
        mlflow.log_param(f"best_{key}", value)


print("\nBest Accuracy :", study.best_value)
print("\nBest Parameters :")
print(study.best_params)


# ----------------------------------------
# Train Final Model
# ----------------------------------------

best_params = study.best_params.copy()

best_params["random_state"] = 42
best_params["n_jobs"] = -1

with mlflow.start_run(run_name="Best Model"):

    model = RandomForestClassifier(**best_params)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=wine.target_names,
        yticklabels=wine.target_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("Confusion-Matrix.png")

    mlflow.log_artifact("Confusion-Matrix.png")

    # Log source code (works only when running as a .py file)
    # mlflow.log_artifact(__file__)

    mlflow.set_tags({
        "Author": "Kiran",
        "Project": "Wine Classification using Optuna + MLflow"
    })

    mlflow.sklearn.log_model(
        model,
        artifact_path="RandomForestModel"
    )

print(f"\nFinal Model Accuracy : {accuracy:.4f}")