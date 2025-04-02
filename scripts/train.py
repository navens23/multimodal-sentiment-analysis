import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPProcessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import AmazonReviewDataset
from models import MultimodalSentimentModel

# Tokenizer and processor
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=clip_processor.image_mean, std=clip_processor.image_std), 
])

# Data loading
try:
    df = pd.read_csv('data/amazon_reviews.csv')  
except FileNotFoundError:
    print("Error: 'amazon_reviews.csv' not found. Please provide the correct path to your dataset.")
    exit()

if not all(col in df.columns for col in ['text', 'image_path', 'label']):
    print("Error: The CSV file must contain 'text', 'image_path', and 'label' columns.")
    exit()

# Handle Label Mapping using LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
num_labels = len(le.classes_)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42) # Added random_state for reproducibility

train_dataset = AmazonReviewDataset(train_df, 'text', 'image_path', 'label', tokenizer, transform)
val_dataset = AmazonReviewDataset(val_df, 'text', 'image_path', 'label', tokenizer, transform)

# Filter out None values (skipped samples) from the datasets
train_dataset.dataframe = train_dataset.dataframe.dropna(subset=['label', 'image_path', 'text']).reset_index(drop=True)
val_dataset.dataframe = val_dataset.dataframe.dropna(subset=['label', 'image_path', 'text']).reset_index(drop=True)

def collate_fn(batch):
    """
    Collates data samples into a batch, filtering out None values (skipped samples).

    Args:
        batch (list): List of samples, where each sample is a tuple of (text_inputs, image, label, original_image).

    Returns:
        tuple: A tuple containing batched text inputs, images, labels, and original images, or (None, None, None, None) if the batch is empty after filtering.
    """
    batch = list(filter(lambda item: item is not None, batch)) # Filter out None from skipped samples
    if not batch:
        return None, None, None, None
    text_inputs = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    original_images = [item[3] for item in batch]
    text_inputs = {key: torch.cat([d[key] for d in text_inputs]) for key in text_inputs[0]}
    return text_inputs, images, labels, original_images

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Model Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalSentimentModel(num_labels=num_labels).to(device)

# Optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Train loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        if batch is None:
            continue # Skip empty batches
        text_inputs, images, labels, _ = batch 
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}  
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(text_inputs, images)
        logits = outputs

        # Calculate loss and backprop
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}')

    # Evaluation after each epoch
    model.eval()
    correct_preds = 0
    total_preds = 0
    all_preds_val = []
    all_labels_val = []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            text_inputs_val, images_val, labels_val, _ = batch 
            text_inputs_val = {key: value.to(device) for key, value in text_inputs_val.items()}
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)

            outputs_val = model(text_inputs_val, images_val)
            logits_val = outputs_val
            _, preds_val = torch.max(logits_val, dim=1)

            correct_preds += (preds_val == labels_val).sum().item()
            total_preds += labels_val.size(0)
            all_preds_val.extend(preds_val.cpu().numpy())
            all_labels_val.extend(labels_val.cpu().numpy())

    accuracy_val = correct_preds / total_preds
    precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(all_labels_val, all_preds_val, average='weighted', zero_division=0)
    print(f'Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {accuracy_val * 100:.2f}%')
    print(f'Epoch {epoch+1}/{num_epochs} - Validation Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1: {f1_val:.4f}')
    model.train() 

# Final Evaluation with Attention Visualization
model.eval()
correct_preds = 0
total_preds = 0
all_preds = []
all_labels = []
visualization_count = 5 # Visualize attention for a few examples

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if batch is None:
            continue 
        text_inputs, images, labels, original_images = batch
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(text_inputs, images)
        logits = outputs
        attention_weights = model.cross_attention.attention_weights.cpu().numpy()

        # Predict the sentiment
        _, preds = torch.max(logits, dim=1)

        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Visualize attention 
        if i < visualization_count:
            for j in range(len(text_inputs['input_ids'])):
                text = tokenizer.decode(text_inputs['input_ids'][j], skip_special_tokens=True)
                original_image = original_images[j]
                attention = attention_weights[j][0][0] # Extract the single attention weight

                plt.figure()
                plt.imshow(original_image)
                plt.title(f'Text: "{text[:50]}..."\nPredicted: {le.inverse_transform([preds[j].item()])[0]}, True: {le.inverse_transform([labels[j].item()])[0]}\nAttention Weight: {attention:.4f}')
                plt.axis('off')
                plt.show()

accuracy = correct_preds / total_preds
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
print(f'\nFinal Validation Accuracy: {accuracy * 100:.2f}%')
print(f'Final Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
