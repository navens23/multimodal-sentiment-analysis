import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, CLIPModel

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism to allow image features to attend to text features.

    Args:
        text_feature_dim (int): Dimension of the text features.
        image_feature_dim (int): Dimension of the image features.
        attention_dim (int): Dimension of the attention space.
    """
    def __init__(self, text_feature_dim, image_feature_dim, attention_dim):
        super(CrossModalAttention, self).__init__()
        self.query_transform = nn.Linear(text_feature_dim, attention_dim)
        self.key_transform = nn.Linear(image_feature_dim, attention_dim)
        self.value_transform = nn.Linear(image_feature_dim, image_feature_dim)
        self.attention_dropout = nn.Dropout(0.1)

    def forward(self, text_features, image_features):
        """
        Forward pass of the cross-modal attention layer.

        Args:
            text_features (torch.Tensor): Tensor of text features (batch_size, text_feature_dim).
            image_features (torch.Tensor): Tensor of image features (batch_size, image_feature_dim).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Tensor of attended image features (batch_size, image_feature_dim).
                - torch.Tensor: Tensor of attention weights (batch_size, 1, 1).
        """
        query = self.query_transform(text_features)  
        key = self.key_transform(image_features)    
        value = self.value_transform(image_features)  

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5) 

        attention_weights = torch.softmax(attention_scores, dim=-1) 

        attention_weights = self.attention_dropout(attention_weights)

        attended_image_features = torch.matmul(attention_weights, value.unsqueeze(1)).squeeze(1) 

        return attended_image_features, attention_weights

class MultimodalSentimentModel(nn.Module):
    """
    Multimodal sentiment analysis model combining BERT for text and CLIP for images with cross-modal attention.

    Args:
        text_model_name (str): Name of the pretrained BERT model to use.
        image_model_name (str): Name of the pretrained CLIP model to use.
        num_labels (int): Number of output classes for sentiment classification.
        attention_dim (int): Dimension of the attention space in the CrossModalAttention layer.
    """
    def __init__(self, text_model_name='bert-base-uncased', image_model_name='openai/clip-vit-base-patch32', num_labels=None, attention_dim=512):
        super(MultimodalSentimentModel, self).__init__()
        self.text_model = BertForSequenceClassification.from_pretrained(text_model_name, num_labels=num_labels)
        self.clip_model = CLIPModel.from_pretrained(image_model_name)
        self.cross_attention = CrossModalAttention(self.text_model.config.hidden_size, self.clip_model.projection_dim, attention_dim)
        self.classifier = nn.Linear(self.text_model.config.hidden_size + self.clip_model.projection_dim, num_labels) 
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_inputs, images):
        """
        Forward pass of the multimodal sentiment analysis model.

        Args:
            text_inputs (dict): Dictionary of input tensors for the text model.
            images (torch.Tensor): Tensor of input images (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Tensor of output logits (batch_size, num_labels).
        """
        text_outputs = self.text_model(**text_inputs, output_hidden_states=True)
        text_features = text_outputs.pooler_output

        image_features = self.clip_model.get_image_features(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) 

        attended_image_features, attention_weights = self.cross_attention(text_features, image_features)
        self.attention_weights = attention_weights # Store for visualization

        combined_features = torch.cat((text_features, attended_image_features), dim=1)
        combined_features = self.dropout(combined_features)

        logits = self.classifier(combined_features)
        return logits
