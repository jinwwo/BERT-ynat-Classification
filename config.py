from transformers import TrainingArguments

def get_training_arguments():
    return TrainingArguments(
        output_dir='results',
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        logging_dir='logs',
        evaluation_strategy='epoch',
        logging_steps=10,
        report_to="tensorboard",
    )
