import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the cleaned census data from the root directory
project_path = os.getcwd()
data_path = os.path.join(project_path, "cleaned_census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Split the provided data to have a train dataset and a test dataset
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
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

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Make sure the model directory exists
os.makedirs(os.path.join(project_path, "model"), exist_ok=True)

# Save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
print(f"Model saved to {model_path}")

encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
print(f"Encoder saved to {encoder_path}")

# Save the label binarizer if it exists
if lb is not None:
    lb_path = os.path.join(project_path, "model", "lb.pkl")
    save_model(lb, lb_path)
    print(f"Label binarizer saved to {lb_path}")

# Load the model
model = load_model(model_path)
print(f"Loading model from {model_path}")

# Run model inferences on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices
# First, clear the existing slice_output.txt file if it exists
with open("slice_output.txt", "w") as f:
    f.write("Model Performance on Data Slices\n")
    f.write("===============================\n\n")

# Convert y_test and preds to numpy arrays for easier indexing
y_test = np.array(y_test)
preds = np.array(preds)


# Helper function to directly compute metrics on slices
def compute_slice_metrics(y_true, y_pred, mask):
    """Compute metrics on a slice of the data"""
    y_slice = y_true[mask]
    preds_slice = y_pred[mask]

    if len(y_slice) == 0 or len(np.unique(y_slice)) < 2:
        return 0.0, 0.0, 0.0

    return compute_model_metrics(y_slice, preds_slice)


# Process each categorical feature
for col in cat_features:
    print(f"Processing slices for feature: {col}")

    with open("slice_output.txt", "a") as f:
        f.write(f"\nFeature: {col}\n")
        f.write("-" * (len(col) + 10) + "\n")

    # Get unique values for the feature
    unique_values = sorted(test[col].unique())

    # Process each unique value
    for value in unique_values:
        # Create a mask for the current value
        mask = test[col] == value
        count = mask.sum()

        # Skip if there are too few samples
        if count < 5:
            p, r, fb = 0, 0, 0
            with open("slice_output.txt", "a") as f:
                f.write(f"{col}: {value}, Count: {count} (too few samples)\n")
                f.write(f"Precision: N/A | Recall: N/A | F1: N/A\n\n")
            continue

        # Calculate metrics for this slice
        try:
            p, r, fb = compute_slice_metrics(y_test, preds, mask)

            with open("slice_output.txt", "a") as f:
                f.write(f"{col}: {value}, Count: {count:,}\n")
                f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n\n")

            print(f"  {col}: {value}, Count: {count:,}")
            print(f"  Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")
        except Exception as e:
            print(f"Error processing slice {col}={value}: {e}")
            with open("slice_output.txt", "a") as f:
                f.write(f"{col}: {value}, Count: {count:,}\n")
                f.write(f"Error calculating metrics: {str(e)}\n\n")

print("Slice performance calculation complete. Results saved to slice_output.txt")