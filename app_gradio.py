import gradio as gr
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pii_masker import mask_pii
from utils import label_mapping

# Model path (should match your folder name)
model_path = "email_classifier_model"

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Inference function
def classify_email(text):
    masked_text, _ = mask_pii(text)  # second value is entities list, not needed here
    inputs = tokenizer(masked_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    label = label_mapping.get(pred, "Unknown")
    return f"Prediction: {label}\n\nMasked Email:\n{masked_text}"

# Gradio interface
iface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=12, label="Paste your email below:"),
    outputs="text",
    title="Email Classification with PII Masking",
    description="Classifies email as Incident, Request, Problem, or Change. PII is masked before classification."
)

if __name__ == "__main__":
    iface.launch()

