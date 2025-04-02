import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Args:
        X_train : Features dataset used for training
        y_train : Labels dataset used for training

    Returns:
        model : Trained machine learning model
    """
    # Initialize a RandomForest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Computes the metrics for the model.

    Args:
        y : Known labels
        preds : Predicted labels

    Returns:
        precision : Precision score
        recall : Recall score
        fbeta : F1 score
    """
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)

    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Args:
        model : Trained machine learning model
        X : Data used for prediction

    Returns:
        preds : Predictions from the model
    """
    return model.predict(X)


def save_model(model, model_path):
    """
    Saves a model to disk.

    Args:
        model : The model to save
        model_path : Path to save the model
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path):
    """
    Loads a model from disk.

    Args:
        model_path : Path to the saved model

    Returns:
        model : The loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def performance_on_categorical_slice(test, y, preds, feature, value):
    """
    Computes performance metrics on a slice of data.

    Args:
        test : Test dataframe
        y : True labels
        preds : Predicted labels
        feature : Feature name to slice on
        value : Feature value to slice on

    Returns:
        precision : Precision score on the slice
        recall : Recall score on the slice
        fbeta : F1 score on the slice
    """
    # Create a mask for the slice
    mask = test[feature] == value

    # Filter the true labels and predictions using the mask
    y_slice = y[mask]
    preds_slice = preds[mask]

    # If the slice is empty, return zeros
    if len(y_slice) == 0:
        return 0, 0, 0

    # Compute metrics on the slice
    return compute_model_metrics(y_slice, preds_slice)