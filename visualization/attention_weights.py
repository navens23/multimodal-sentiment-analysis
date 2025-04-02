import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from models import MultimodalSentimentModel
from dataset import AmazonReviewDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = MultimodalSentimentModel(num_labels=3)  
model.load_state_dict(torch.load('models/model_checkpoint.pth')) 
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
df = pd.read_csv('data/amazon_reviews.csv')
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
dataset = AmazonReviewDataset(df, 'text', 'image_path', 'label', tokenizer, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Visualize attention weights
def visualize_attention(text, image, attention_weight, label, prediction):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.barh([text], [attention_weight])
    plt.xlabel('Attention Weight')
    plt.title(f'True Label: {label}, Prediction: {prediction}')
    plt.show()

# Iterate through the dataloader and visualize
with torch.no_grad():
    for i, (text_inputs, images, labels, original_images) in enumerate(dataloader):
        if i >= 5:  # Visualize 5 examples
            break
        text_inputs = {key: value for key, value in text_inputs.items()}
        outputs = model(text_inputs, images)
        _, preds = torch.max(outputs, dim=1)
        attention_weights = model.cross_attention.attention_weights.cpu().numpy()
        visualize_attention(tokenizer.decode(text_inputs['input_ids'][0], skip_special_tokens=True), original_images[0], attention_weights[0][0][0], labels[0].item(), preds[0].item())
