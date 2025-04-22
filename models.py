from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)

        }

    def __len__(self):
        return len(self.labels)

def prepare_data(df):
    texts = df["email"].tolist()
    le = LabelEncoder()
    labels = le.fit_transform(df["type"])
    return texts, labels, le

def train_model(df):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    texts, labels, le = prepare_data(df)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_dataset = EmailDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmailDataset(val_texts, val_labels, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(set(labels))
    )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        logging_dir="./logs",
        save_total_limit=1,
        save_strategy="epoch",
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained("email_classifier_model")
    tokenizer.save_pretrained("email_classifier_model")
    print("✅ Model training completed and saved.")


