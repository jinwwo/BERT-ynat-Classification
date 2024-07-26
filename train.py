from transformers import Trainer, DataCollatorWithPadding
from config import get_training_arguments
from dataset import load_and_tokenize_dataset
from utils import compute_metrics, seed_everything, plot_confusion_matrix
from transformers import AutoModelForSequenceClassification
from callback import LossCallback


def train(output_dir='results'):
    seed_everything()
    datasets, tokenizer = load_and_tokenize_dataset()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'klue/bert-base', num_labels=datasets['train'].features['label'].num_classes
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = get_training_arguments()
    loss_callback = LossCallback(logging_interval=30)
    
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
    loss_callback.plot_loss()
    
    results = trainer.evaluate()
    predictions = trainer.predict(datasets["test"])
    plot_confusion_matrix(predictions)
    
    
if __name__ == "__main__":
    train()