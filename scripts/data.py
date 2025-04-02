import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class AmazonReviewDataset(Dataset):
    """
    Dataset class for loading Amazon reviews with text and images.

    Args:
        dataframe (pd.DataFrame): DataFrame containing review data with text, image paths, and labels.
        text_column (str): Name of the column containing review text.
        image_column (str): Name of the column containing image file paths.
        label_column (str): Name of the column containing review labels.
        tokenizer (transformers.BertTokenizer): Tokenizer for processing text.
        transform (torchvision.transforms): Image transformations to apply.
    """
    def __init__(self, dataframe, text_column, image_column, label_column, tokenizer, transform=None):
        self.dataframe = dataframe
        self.text_column = text_column
        self.image_column = image_column
        self.label_column = label_column
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx][self.text_column]
        image_path = self.dataframe.iloc[idx][self.image_column]
        label = self.dataframe.iloc[idx][self.label_column]

        # Process the text
        text_inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

        # Process the image
        try:
            original_image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(original_image)
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Skipping this sample.")
            return None, None, None, None 

        return text_inputs, image, label, original_image
