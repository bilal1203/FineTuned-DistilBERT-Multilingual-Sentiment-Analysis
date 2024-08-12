from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

def load_model_and_tokenizer(model_name="distilbert-base-multilingual-cased", num_labels=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    return model, tokenizer

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)

def prepare_datasets(train_dataset, val_dataset, test_dataset, tokenizer):
    tokenized_train = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_test = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    
    return tokenized_train, tokenized_val, tokenized_test

def get_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)