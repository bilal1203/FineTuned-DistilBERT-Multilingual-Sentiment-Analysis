import torch
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from typing import List

def evaluate_model(trainer: Trainer, test_dataset: Dataset) -> None:
    try:
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate model: {e}")

def predict_examples(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, examples: List[str]) -> None:
    try:
        inputs = tokenizer(examples, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)

        for example, prediction in zip(examples, predictions):
            print(f"Text: {example}")
            print(f"Predicted rating: {prediction.item() + 1} stars\n")
    except Exception as e:
        raise RuntimeError(f"Failed to predict examples: {e}")

def get_top_words(text: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, top_k: int = 5) -> List[str]:
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        attention = outputs.attentions[-1].mean(dim=1).mean(dim=1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_attention = list(zip(tokens, attention[0].tolist()))
        top_tokens = sorted(token_attention, key=lambda x: x[1], reverse=True)[:top_k]
        
        return [token for token, _ in top_tokens if token not in ["[CLS]", "[SEP]", "[PAD]"]]
    except Exception as e:
        raise RuntimeError(f"Failed to get top words: {e}")

def run_evaluation(trainer: Trainer, test_dataset: Dataset) -> None:
    evaluate_model(trainer, test_dataset)
    
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    examples = [
        "This product is amazing! I love it.",
        "Terrible experience. Would not recommend.",
        "It's okay, nothing special.",
        "This product is great!",
        "I don't like this product.",
        "It's an okay product.",
        "I love this product!",
        "I'm not sure about this product.",
    ]
    predict_examples(model, tokenizer, examples)
    
    text = "This product is amazing! I love it."
    top_words = get_top_words(text, model, tokenizer)
    print(f"Top influential words: {top_words}")

if __name__ == "__main__":
    from src.train import train_model
    try:
        trainer, test_dataset = train_model()
        run_evaluation(trainer, test_dataset)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")