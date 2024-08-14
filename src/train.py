from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
from data_preparation import load_and_preprocess_data
from model import load_model_and_tokenizer, prepare_datasets, get_data_collator
from typing import Dict, Tuple

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

def get_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=5e-5,
        optimizer="AdamW",
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,  # Keep only the 3 best models
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # You can change this to "accuracy" if preferred
    )

def train_model(training_args=None):
    try:
        print("Loading and preprocessing data...")
        train_dataset, val_dataset, test_dataset = load_and_preprocess_data()
        
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer()
        
        print("Preparing datasets...")
        tokenized_train, tokenized_val, tokenized_test = prepare_datasets(train_dataset, val_dataset, test_dataset, tokenizer)
        data_collator = get_data_collator(tokenizer)

        if training_args is None:
            training_args = get_training_args()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("Starting training...")
        trainer.train()

        return trainer, tokenized_test

    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise

if __name__ == "__main__":
    train_model()