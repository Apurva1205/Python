# XGBoost Classifier Example
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def data_handling(data: dict) -> tuple:
    # Split dataset into features and target
    # data is features
    return (data["data"], data["target"])


def xgboost(features: np.ndarray, target: np.ndarray) -> XGBClassifier:
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Define the XGBoost classifier and the grid of hyperparameters to search
    xgb = XGBClassifier()
    param_grid = {
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5],
        "gamma": [0, 0.1, 0.2],
    }

    # Perform a grid search with cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(
        xgb, param_grid, scoring="accuracy", cv=5, n_jobs=-1, verbose=1
    )
    grid_search.fit(features_scaled, target)

    # Print the best hyperparameters and the corresponding accuracy score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best accuracy score:", grid_search.best_score_)

    # Return the XGBoost classifier with the best hyperparameters
    return grid_search.best_estimator_


def main() -> None:
    # Load the Iris dataset
    iris = load_iris()
    features, targets = data_handling(iris)

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, random_state=42
    )

    # Create an XGBoost classifier with the best hyperparameters found by grid search
    xgboost_classifier = xgboost(x_train, y_train)

    # Display the confusion matrix of the classifier with both training and test sets
    names = iris["target_names"]
    ConfusionMatrixDisplay.from_estimator(
        xgboost_classifier,
        x_test,
        y_test,
        display_labels=names,
        cmap="Blues",
        normalize="true",
    )
    plt.title("Normalized Confusion Matrix - IRIS Dataset")
    plt.show()


if __name__ == "__main__":
    main()
