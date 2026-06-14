import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = os.environ.get('HF_MODEL_NAME', 'jayagovind1703/my-mlops-model')
input_text = os.environ.get('INPUT_TEXT', 'This movie was fantastic!')
hf_token = os.environ.get('HF_TOKEN')

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)

inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

if "token_type_ids" in inputs:
    del inputs["token_type_ids"]

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class_id = logits.argmax().item()
label = model.config.id2label[predicted_class_id]
score = torch.softmax(logits, dim=1)[0][predicted_class_id].item()

print(f'Input: {input_text}')
print(f"Result: [{{'label': '{label}', 'score': {score:.4f}}}]")