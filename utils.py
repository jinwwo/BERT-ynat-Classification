import os
import random
from collections import Counter
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from transformers.trainer_utils import PredictionOutput

label_map = {
    "LABEL_0": "IT",
    "LABEL_1": "Economy",
    "LABEL_2": "Society",
    "LABEL_3": "life&culture",
    "LABEL_4": "World",
    "LABEL_5": "Sports",
    "LABEL_6": "Politics",
}


def seed_everything(seed_value: Optional[int]) -> None:
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(p: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def plot_confusion_matrix(predictions: PredictionOutput, output=Optional[str]) -> None:
    if not os.path.splitext(output)[1]:
            output += ".png"
    print(predictions.metrics)
    preds = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(predictions.label_ids, preds)
    labels = [label_map[f"LABEL_{i}"] for i in range(len(label_map))]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix with Label Names")
    plt.savefig(output)
    plt.close()


def under_sampling(dataset: DatasetDict, seed: Optional[int] = 42) -> Dataset:
    labels = dataset["train"]["label"]
    min_val = min(Counter(labels).values())
    ids_per_classes = {
        label: np.where(labels == label)[0] for label in np.unique(labels)
    }
    undersampled = []

    for _, ids in ids_per_classes.items():
        if len(ids) > min_val:
            undersampled.extend(np.random.choice(ids, min_val, replace=False))
        else:
            undersampled.extend(ids)

    return dataset["train"].select(undersampled).shuffle(seed=seed)