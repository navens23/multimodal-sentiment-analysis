# Disclaimer

Please note that the code presented in this repository constitutes a specific component from a broader project. It is provided solely for illustrative and PhD application-focused purposes. The complete project encompasses a wider range of functionalities and modules beyond what is currently displayed here due to privacy considerations.


# Multimodal Sentiment Analysis

This Repository contains code for a multimodal sentiment analysis model that combines BERT for text and CLIP for images with cross-modal attention. The model is trained and evaluated on Amazon reviews with both text and images. 

This is an implementation of Cross-modal attention that serves as an advanced fusion method, implementing attention mechanisms that allow interaction between the text and image modalities before the final classification layer.


## Repository Structure

- `data/`: Directory to store the dataset (e.g., `amazon_reviews.csv`).
- `scripts/`: Directory containing the main scripts for dataset handling, model definition, and training.
  - `dataset.py`: Script for the `AmazonReviewDataset` class.
  - `models.py`: Script for the model definitions including cross-modal attention and multimodal sentiment model.
  - `train.py`: Script for training and evaluating the model.
- `notebooks/`: Directory containing Jupyter notebooks for visualization and analysis.
  - `attention_weights.py`: Python for visualizing attention weights and model predictions.

## Setup

### Requirements

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### Dataset
Please be advised that due to privacy considerations, the model weights and the original dataset used in this project have been removed from this repository. I am unable to provide access to this sensitive information.

## Usage

### Training

Run the `train.py` script to train the model:

```bash
python scripts/train.py
```

### Visualization

Use the `attention_weights.py` python to visualize attention weights and model predictions.

## License

This project is licensed under the MIT License.
