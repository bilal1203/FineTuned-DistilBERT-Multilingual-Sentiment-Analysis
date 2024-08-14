from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from typing import Tuple, Dict, Any

def load_model_and_tokenizer(model_name: str = "distilbert-base-multilingual-cased", num_labels: int = 5) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        return model, tokenizer
    except Exception as e:
        if "Error: No such file or directory" in str(e):
            raise RuntimeError(f"Model or tokenizer not found: {model_name}. Make sure to download the model or provide a valid model name.")
        else:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")

def prepare_datasets(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    tokenize = lambda examples: tokenizer(examples["text"], truncation=True)
    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_val = val_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)
    
    return tokenized_train, tokenized_val, tokenized_test

def get_data_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)

if __name__ == "__main__":
    # You can add some example usage or tests here if needed
    pass