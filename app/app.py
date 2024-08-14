import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the fine-tuned model and tokenizer
model_path = "../fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    
    confidence = probabilities[0][prediction].item()
    
    sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    predicted_sentiment = sentiment_labels[prediction]
    
    return predicted_sentiment, confidence

def gradio_interface(text):
    sentiment, confidence = predict_sentiment(text)
    return f"Sentiment: {sentiment}\nConfidence: {confidence:.2f}"

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=4, placeholder="Enter text here..."),
    outputs="text",
    title="Multilingual Sentiment Analysis",
    description="Enter a review in any language to predict its sentiment."
)

# Launch the interface
iface.launch()