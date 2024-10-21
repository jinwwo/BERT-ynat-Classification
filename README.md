## Classification with BERT
This is a project that performs classification by fine-tuning BERT (Bidirectional Encoder Representations from Transformers)

## Features
- Load and preprocess [YNAT](https://huggingface.co/datasets/klue/klue/viewer/ynat) dataset
- Text Classification using BERT model
- Training and evaluation
- Confustion matrix, loss plot visualization
  
## Datasets
- Source: [klue-ynat](https://huggingface.co/datasets/klue/klue/viewer/ynat)

## Preprocessing
- Extract "title" and "label" columns (Removed "guid", "url", "date")
- Due to class imbalance, undersampling was performed based on the class with the fewest instances.

## Train model
```bash
python train.py --seed 42 --epoch 3 --model klue/bert-base --batch_size 32
```

## Training config
- base model: [klue/bert-base](https://huggingface.co/klue/bert-base)
- epochs: 3
- learning rate: 5e-5
- scheduler: cosine
- weight decay: 0.01
- batch size: 32

## Performance Metrics
|Metric|Value|
|---|---|
|Accuracy|86.28|
|F1 Score|86.21|
|Precision|86.26|
|Recall|86.28|

