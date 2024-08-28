import argparse
import os

from transformers import (AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer)

from callback import LossCallback
from config import get_training_arguments
from dataset import load_and_tokenize_dataset
from utils import compute_metrics, plot_confusion_matrix, seed_everything


def train(output_dir='results'):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--epoch", type=int, required=False, default=3)
    parser.add_argument("--model", type=str, required=False, default="klue/bert-base")
    parser.add_argument("--plot_name", type=str, required=False, default='plot')
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--result_name", type=str, required=False, default='result')
    parser.add_argument("--logging_interval", type=int, required=False, default=30)

    args = parser.parse_args()

    seed_everything(args.seed)

    base_dir = os.path.join(os.getcwd(), 'plots')
    loss_plot_dir = os.path.abspath(os.path.join(base_dir, 'loss'))
    cfm_dir = os.path.abspath(os.path.join(base_dir, 'cfm'))
    result_dir = os.path.abspath(os.path.join(base_dir, 'result', args.result_name))

    os.makedirs(loss_plot_dir, exist_ok=True)
    os.makedirs(cfm_dir, exist_ok=True)

    datasets, tokenizer = load_and_tokenize_dataset(tokenizer=args.model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loss_callback = LossCallback(logging_interval=args.logging_interval)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=datasets['train'].features['label'].num_classes
    )
    training_args = get_training_arguments(
        output_dir=result_dir, epochs=args.epoch, batch_size=args.batch_size
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback]
    )
    
    trainer.train()
    loss_callback.plot_loss(loss_plot_dir + '/loss_')
    
    results = trainer.evaluate()
    predictions = trainer.predict(datasets["test"])
    plot_confusion_matrix(predictions, output=cfm_dir + '/cfm_')
    
    
if __name__ == "__main__":
    train()