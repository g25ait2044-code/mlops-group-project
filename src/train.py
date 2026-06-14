import json
import os
import pandas as pd
import wandb
from datasets import Dataset
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

# 1. Load Secrets
secrets = UserSecretsClient()
os.environ['WANDB_API_KEY'] = secrets.get_secret('WANDB_API_KEY')
login(token=secrets.get_secret('HF_TOKEN'))
wandb.login()

# Load labels
with open('id2label.json') as f:
    id2label = json.load(f)
label2id = {v: int(k) for k, v in id2label.items()}

# Load DistilBERT tokenizer and model
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

# 2. Initialize W&B Run
wandb.init(
    project='mlops-assignment3',
    name='run-v1',  # Change this to run-v2 for your second experiment
    config={
        'model': 'distilbert-base-uncased',
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 3e-5,  # Change this hyperparameter in run 2
        'version': 'v1',
        'platform': 'Kaggle',
    }
)

# 1. Load the cleaned CSV files into Pandas DataFrames
# (Adjust the path if your CSVs are stored in a specific Kaggle input directory)
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Convert DataFrames to Hugging Face Dataset objects
train_dataset_raw = Dataset.from_pandas(train_df)
test_dataset_raw = Dataset.from_pandas(test_df)


# 3. Define the tokenization behavior
# This applies the DistilBERT tokenizer you loaded earlier to the 'text' column
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=512
    )


# 4. Map the tokenization function across the entire dataset
train_dataset = train_dataset_raw.map(tokenize_function, batched=True)
test_dataset = test_dataset_raw.map(tokenize_function, batched=True)

# Optional: Remove the raw text column as the model only needs the tokens and labels
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])


# 3. Define Metrics & Train
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted'),
    }


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to='wandb',
    run_name='run-v1',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 4. Push Best Model to Hugging Face Hub
model.push_to_hub('jayagovind1703/my-mlops-model')
tokenizer.push_to_hub('jayagovind1703/my-mlops-model')

# Log HF URL to W&B
wandb.run.summary['huggingface_model'] = 'https://huggingface.co/jayagovind1703/my-mlops-model'

wandb.finish()

