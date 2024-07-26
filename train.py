from transformers import Trainer, DataCollatorWithPadding
from config import get_training_arguments
from dataset import load_and_tokenize_dataset
from utils import compute_metrics, seed_everything, plot_confusion_matrix
from transformers import AutoModelForSequenceClassification
from callback import LossCallback

if __name__ == "__main__":
    seed_everything()
    tokenized_datasets, tokenizer = load_and_tokenize_dataset()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        'klue/bert-base', num_labels=dataset['train'].features['label'].num_classes
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = get_training_arguments()
    loss_callback = LossCallback(logging_interval=30)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[loss_callback]
    )
    
    trainer.train()
    loss_callback.plot_loss()
    
    model.save_pretrained('bert-ynat-cls')
    tokenizer.save_pretrained('bert-ynat-cls')
    
    results = trainer.evaluate()
    predictions = trainer.predict(tokenized_datasets["test"])
    plot_confusion_matrix(predictions)