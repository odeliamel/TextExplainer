# Explaining high-dimensional text classifiers
by Odelia Melamed, Rich Caruana

Full paper: https://arxiv.org/abs/2311.13454 

This project provides a library for explainability in sentiment analysis models, and example focusing on IMDB movie reviews. It includes tools for evaluating models, generating explanations for predictions, and visualizing insights.


## Features

- **Sentiment Analysis Model**: Implements a recurrent neural network (`SentimentRNN`) for sentiment classification.
- **Explainability**: Provides the `HD_Explainer` class to explain model predictions using gradients and high-dimensional analysis.
- **Visualization**: Includes tools for visualizing gradient norms, inner products, and other metrics.
- **Preprocessing**: Utilities for tokenizing text and preparing data for model input.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/explainabilitylib.git
   cd explainabilitylib
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the IMDB dataset and pre-trained weights (already included in `IMDB_weights/`).

## Usage

### Running the Jupyter Notebook

The main workflow is demonstrated in the `imdb.ipynb` notebook. Open it in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook imdb.ipynb
```

The prediction explanation is presented visually, where the different words are colored according to thier importance (with strong colors means strong explanation score).

For the following example the given prediction is a **negative sentiment**, and the resulted explanation is: 

![image](https://github.com/user-attachments/assets/5c9ec171-6221-453b-a750-fe0231c19e90)


### Key Steps in the Notebook

1. **Load Data and Model**:
   - Load the IMDB dataset and pre-trained `SentimentRNN` model.
   - Initialize the `Classifier` and `HD_Explainer`.

2. **Train/Test Split**:
   - Split the dataset into training and testing sets.

3. **Explain Predictions**:
   - Use the `HD_Explainer` to generate explanations for model predictions.

4. **Visualize Insights**:
   - Plot gradient norms and other metrics to understand model behavior.

### Example Code

```python
from HD_explainer import HD_Explainer
from classifier import Classifier
from imdb_model import SentimentRNN

# Load model and data
model = SentimentRNN(no_layers=2, vocab=vocab, vocab_size=len(vocab)+1, hidden_dim=256, embedding_dim=64)
model.load_state_dict(torch.load('IMDB_weights/state_dict-50epochs-0.pt'))
classifier = Classifier(text_to_tokens=model.tokenize, embedding=model.embedding, model=model)

# Initialize explainer
explainer = HD_Explainer(classifier, surrogate_models=[], token_split=lambda x: [text.split() for text in x], vocab=vocab, max_len=500)

# Explain a prediction
inputs = ["This movie was fantastic!"]
labels = torch.tensor([1])  # Positive sentiment
explainer.explain(inputs, labels)
```

## Pre-trained Models

Pre-trained models are stored in the `IMDB_weights/` directory. These include weights for the `SentimentRNN` model trained for 50 epochs.



## Usage

### Running the Jupyter Notebook

The main workflow is demonstrated in the `imdb.ipynb` notebook. Open it in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook imdb.ipynb
```

### Key Steps in the Notebook

1. **Load Data and Model**:
   - Load the IMDB dataset and pre-trained `SentimentRNN` model.
   - Initialize the `Classifier` and `HD_Explainer`.

2. **Train/Test Split**:
   - Split the dataset into training and testing sets.

3. **Explain Predictions**:
   - Use the `HD_Explainer` to generate explanations for model predictions.

4. **Visualize Insights**:
   - Plot gradient norms and angles to understand model behavior.

### Example Code

```python
from HD_explainer import HD_Explainer
from classifier import Classifier
from imdb_model import SentimentRNN
import torch

# Load model and data
model = SentimentRNN(no_layers=2, vocab=vocab, vocab_size=len(vocab)+1, hidden_dim=256, embedding_dim=64)
model.load_state_dict(torch.load('IMDB_weights/state_dict-50epochs-0.pt'))
classifier = Classifier(text_to_tokens=model.tokenize, embedding=model.embedding, model=model)

# Initialize explainer
explainer = HD_Explainer(classifier, surrogate_models=[], token_split=lambda x: [text.split() for text in x], vocab=vocab, max_len=500)

# Explain a prediction
inputs = ["This movie was fantastic!"]
labels = torch.tensor([1])  # Positive sentiment
explainer.explain(inputs, labels)
```

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- tqdm

## Pre-trained Models

Pre-trained models are stored in the `IMDB_weights/` directory. These include weights for the `SentimentRNN` model trained for 50 epochs.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- IMDB dataset for sentiment analysis.
- PyTorch for deep learning framework.
- NLTK for natural language processing utilities.
```
