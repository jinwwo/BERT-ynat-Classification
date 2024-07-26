import numpy as np
import torch
import random
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


label_map = {
    'LABEL_0': 'IT',
    'LABEL_1': 'Economy',
    'LABEL_2': 'Society',
    'LABEL_3': 'life&culture',
    'LABEL_4': 'World',
    'LABEL_5': 'Sports',
    'LABEL_6': 'Politics'
}


def seed_everything(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def plot_confusion_matrix(predictions, output='confusion_matrix.jpg'):
    print(predictions.metrics)
    preds = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(predictions.label_ids, preds)
    labels = [label_map[f'LABEL_{i}'] for i in range(len(label_map))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix with Label Names')
    plt.savefig(output)
    plt.close()


def under_sampling(dataset):
    labels = dataset['train']['label']
    min_val = min(Counter(labels).values())
    ids_per_classes = {label: np.where(labels==label)[0] for label in np.unique(labels)}
    undersampled = []
    
    for _, ids in ids_per_classes.items():
        if len(ids) > min_val:
            undersampled.extend(np.random.choice(ids, min_val, replace=False))
        else:
            undersampled.extend(ids)
    
    return dataset['train'].select(undersampled).shuffle(seed=42)