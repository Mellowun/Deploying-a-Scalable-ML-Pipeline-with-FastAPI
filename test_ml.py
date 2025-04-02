import pytest
import pandas as pd
import numpy as np
from ml.model import train_model, compute_model_metrics, inference, save_model, load_model
from ml.data import process_data, apply_label
import os


@pytest.fixture
def sample_data():
    """
    Create sample data for testing.
    """
    data = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "workclass": ["Private", "Private", "Self-emp", "Federal-gov", "Private"],
        "fnlgt": [100000, 200000, 300000, 400000, 500000],
        "education": ["HS-grad", "Bachelors", "Masters", "Doctorate", "HS-grad"],
        "education-num": [9, 13, 14, 16, 9],
        "marital-status": ["Single", "Married", "Divorced", "Married", "Single"],
        "occupation": ["Craft-repair", "Prof-specialty", "Exec-managerial", "Prof-specialty", "Other-service"],
        "relationship": ["Own-child", "Husband", "Not-in-family", "Husband", "Unmarried"],
        "race": ["White", "White", "Black", "White", "Hispanic"],
        "sex": ["Male", "Male", "Female", "Male", "Female"],
        "capital-gain": [0, 10000, 0, 20000, 0],
        "capital-loss": [0, 0, 5000, 0, 0],
        "hours-per-week": [40, 45, 50, 60, 35],
        "native-country": ["United-States", "United-States", "Cuba", "United-States", "Mexico"],
        "salary": ["<=50K", ">50K", ">50K", ">50K", "<=50K"]
    })
    return data


def test_train_model(sample_data):
    """
    Test if the train_model function returns a model of the expected type.
    """
    # Process the data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Train the model
    model = train_model(X, y)

    # Check if the model has the expected attributes
    assert hasattr(model, 'predict')
    assert hasattr(model, 'fit')


def test_compute_model_metrics():
    """
    Test if compute_model_metrics returns the expected values.
    """
    # Create sample data
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    # Calculate metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Check if the metrics are calculated correctly
    assert precision == 1.0  # 2 true positives, 0 false positives
    assert recall == 0.6666666666666666  # 2 true positives, 1 false negative
    assert fbeta == 0.8  # F1 score


def test_inference(sample_data):
    """
    Test if the inference function returns predictions of the expected shape.
    """
    # Process the data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # Train the model
    model = train_model(X, y)

    # Make predictions
    preds = inference(model, X)

    # Check if predictions have the expected shape and type
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(sample_data),)
    assert np.all(np.isin(preds, [0, 1]))  # Binary classification