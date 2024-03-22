import numpy as np
import pandas as pd
def parse_response(response):
    response = response.lower()

    if any(label in response for label in ["major_decrease", "major decrease"]):
        return "MAJOR_DECREASE"

    if any(label in response for label in ["minor_decrease", "minor decrease"]):
        return "MINOR_DECREASE"

    if any(label in response for label in ["no_change", "no change"]):
        return "NO_CHANGE"

    if any(label in response for label in ["minor_increase", "minor increase"]):
        return "MINOR_INCREASE"

    if any(label in response for label in ["major_increase", "major increase"]):
        return "MAJOR_INCREASE"

    return "UNKNOWN"

def eval_metrics(results):
    results = pd.DataFrame(results, columns=["sentence", "label", "response", "predicted_label"])
    accuracy_total = np.mean(results["label"] == results["predicted_label"])

    # Accuracy for each class
    accuracy_per_class = {
        label: np.mean(
            results[results["label"] == label]["label"] == results[results["label"] == label]["predicted_label"])
        for label in ["MAJOR_DECREASE", "MINOR_DECREASE", "NO_CHANGE", "MINOR_INCREASE", "MAJOR_INCREASE"]
    }

    # Precision for each class
    precision = {
        label: np.mean(
            (results["label"] == label) & (results["predicted_label"] == label)
        ) / np.mean((results["predicted_label"] == label))
        for label in ["MAJOR_DECREASE", "MINOR_DECREASE", "NO_CHANGE", "MINOR_INCREASE", "MAJOR_INCREASE"]
    }

    # Recall for each class
    recall = {
        label: np.mean(
            (results["label"] == label) & (results["predicted_label"] == label)
        ) / np.mean(results["label"] == label)
        for label in ["MAJOR_DECREASE", "MINOR_DECREASE", "NO_CHANGE", "MINOR_INCREASE", "MAJOR_INCREASE"]
    }

    # F1 for each class
    f1 = {
        label: 2 * (precision[label] * recall[label]) / (precision[label] + recall[label])
        for label in ["MAJOR_DECREASE", "MINOR_DECREASE", "NO_CHANGE", "MINOR_INCREASE", "MAJOR_INCREASE"]
    }

    # Replace NaNs with 0
    precision = {label: 0 if np.isnan(precision[label]) else precision[label] for label in precision}
    recall = {label: 0 if np.isnan(recall[label]) else recall[label] for label in recall}
    f1 = {label: 0 if np.isnan(f1[label]) else f1[label] for label in f1}

    return {
        "accuracy_total": accuracy_total,
        "accuracy_per_class": accuracy_per_class,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "percentage_of_each_class_predicted": results["predicted_label"].value_counts(normalize=True).to_dict(),
        "percentage_of_each_class_dataset": results["label"].value_counts(normalize=True).to_dict()
    }
