import torch

def evaluate_model(trainer, test_dataset):
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")

def predict_examples(model, tokenizer, examples):
    inputs = tokenizer(examples, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)

    for example, prediction in zip(examples, predictions):
        print(f"Text: {example}")
        print(f"Predicted rating: {prediction.item() + 1} stars\n")

def get_top_words(text, model, tokenizer, top_k=5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    attention = outputs.attentions[-1].mean(dim=1).mean(dim=1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_attention = list(zip(tokens, attention[0].tolist()))
    top_tokens = sorted(token_attention, key=lambda x: x[1], reverse=True)[:top_k]
    
    return [token for token, _ in top_tokens if token not in ["[CLS]", "[SEP]", "[PAD]"]]

if __name__ == "__main__":
    from train import train_model
    trainer, test_dataset = train_model()
    evaluate_model(trainer, test_dataset)
    
    model = trainer.model
    tokenizer = trainer.tokenizer
    
    examples = [
        "This product is amazing! I love it.",
        "Terrible experience. Would not recommend.",
        "It's okay, nothing special.",
    ]
    predict_examples(model, tokenizer, examples)
    
    text = "This product is amazing! I love it."
    top_words = get_top_words(text, model, tokenizer)
    print(f"Top influential words: {top_words}")