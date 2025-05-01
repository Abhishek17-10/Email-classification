#  Email Classification with PII Masking 

This project classifies emails into 4 categories — **Incident, Request, Problem, or Change** — while protecting privacy by **masking PII (Personally Identifiable Information)** like names, emails, and phone numbers.

Live Demo 👉 [Try it on Hugging Face Spaces 🚀](https://huggingface.co/spaces/Shady2773/email-classifier)

---

## ✨ Features

- ✅ Classifies emails using a DistilBERT model
- ✅ PII Masking using SpaCy (`en_core_web_sm`)
- ✅ Clean and responsive Gradio UI
- ✅ Hosted for free on Hugging Face Spaces
- ✅ Simple to run locally

---

## 🧠 Model Training Info

- ✅ Base Model: `distilbert-base-uncased`
- ✅ Fine-tuned on labeled email data
- ✅ Categories: `Incident`, `Request`, `Problem`, `Change`
- ✅ Class weights handled to reduce bias
- ✅ Trained using `transformers.Trainer`


---

## 🛠️ Tech Stack

| Component      | Description                                |
|----------------|--------------------------------------------|
| 🧠 Model        | `DistilBERT` fine-tuned for classification |
| 🧹 PII Masking  | `SpaCy` with regex + NER                   |
| 🌐 Frontend     | `Gradio` for simple web UI                 |
| ☁️ Deployment   | `Hugging Face Spaces`                      |

---

## 🧪 How to Run Locally

```bash
git clone https://github.com/Abhishek17-10/Email-classification.git
cd Email-classification
pip install -r requirements.txt
python app.py

