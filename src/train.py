from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
from data_preparation import load_and_preprocess_data
from model import load_model_and_tokenizer, prepare_datasets, get_data_collator
from tqdm import tqdm
import time
from typing import Dict

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class TqdmTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        tqdm.write(str(output))

def train_model():
    start_time = time.time()
    
    print("Loading and preprocessing data...")
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data()
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("Preparing datasets...")
    tokenized_train, tokenized_val, tokenized_test = prepare_datasets(train_dataset, val_dataset, test_dataset, tokenizer)
    data_collator = get_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb logging
    )

    trainer = TqdmTrainer(
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
    
    print("Saving model...")
    trainer.save_model("./fine_tuned_model")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return trainer, tokenized_test

if __name__ == "__main__":
    train_model()