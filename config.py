from typing import Optional

from transformers import TrainingArguments


def get_training_arguments(
        output_dir = Optional[str],
        epochs = Optional[int],
        batch_size = Optional[int],
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        logging_dir='logs',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        logging_steps=10,
        report_to="tensorboard",
    )