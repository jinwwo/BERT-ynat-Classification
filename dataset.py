from typing import Optional, Tuple

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from utils import under_sampling

def load_and_tokenize_dataset(
    tokenizer: Optional[str],
    seed: Optional[int],
) -> Tuple[DatasetDict, PreTrainedTokenizerBase]:
    dataset = load_dataset("klue", "ynat")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def tokenize_function(examples):
        return tokenizer(examples["title"], padding=True, truncation=True)
    
    tokenized_datasets = dataset.map(
        function=tokenize_function, batched=True, remove_columns=["guid", "url", "date"]
    )
    validation_test_split = tokenized_datasets["validation"].train_test_split(
        test_size=0.1
    )
    tokenized_datasets = DatasetDict(
        {
            "train": tokenized_datasets["train"],
            "validation": validation_test_split["train"],
            "test": validation_test_split["test"],
        }
    )
    tokenized_datasets["train"] = under_sampling(tokenized_datasets, seed)

    return tokenized_datasets, tokenizer