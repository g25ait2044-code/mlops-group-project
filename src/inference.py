import os
from transformers import pipeline

model_name = os.environ.get('HF_MODEL_NAME', 'your-hf-username/my-mlops-model')
input_text = os.environ.get('INPUT_TEXT', 'This movie was fantastic!')

classifier = pipeline('text-classification', model=model_name, token=os.environ.get('HF_TOKEN'))
result = classifier(input_text)

print(f'Input: {input_text}')
print(f'Result: {result}')