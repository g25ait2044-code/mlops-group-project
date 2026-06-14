from datasets import load_dataset
import json
import pandas as pd

# Load dataset and sample to reduce size [cite: 47]
dataset = load_dataset('imdb')
train_df = pd.DataFrame(dataset['train']).sample(5000, random_state=42)
test_df = pd.DataFrame(dataset['test']).sample(1000, random_state=42)

# Create and save label mapping [cite: 47]
id2label = {0: 'NEGATIVE', 1: 'POSITIVE'}
with open('id2label.json', 'w') as f:
    json.dump(id2label, f)

# Save data locally [cite: 47]
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
print('Done! Rows:', len(train_df))