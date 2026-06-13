# IMDB Sentiment Analysis using LSTM

## Project Overview

This project uses a Long Short-Term Memory (LSTM) neural network to classify movie reviews as positive or negative.

The goal is to learn sequence modeling, word embeddings, and deep learning for Natural Language Processing (NLP).

## Dataset

IMDB Movie Review Dataset

* 25,000 training reviews
* 25,000 testing reviews

Target:

* 0 = Negative Review
* 1 = Positive Review

## Data Preprocessing

Steps performed:

1. Loaded IMDB dataset from TensorFlow/Keras
2. Limited vocabulary to the 10,000 most frequent words
3. Converted reviews into token sequences
4. Applied sequence padding with max length of 200

## Model Architecture

* Embedding Layer (10000, 32)
* LSTM Layer (32 units)
* Dense Layer (1 neuron, Sigmoid)

## Model Compilation

Optimizer:

* Adam

Loss Function:

* Binary Crossentropy

Metric:

* Accuracy

## Results

Training Accuracy:

* 95.8%

Validation Accuracy:

* 86.5%

Test Accuracy:

* 85.9%

## Key Learning Outcomes

* NLP Fundamentals
* Tokenization
* Sequence Data
* Padding
* Embedding Layers
* LSTM Networks
* Binary Classification
* Overfitting Analysis
* Deep Learning Workflow

## Important Concepts Learned

Text cannot be directly processed by neural networks.

Pipeline used:

Text
→ Token IDs
→ Padding
→ Embedding
→ LSTM
→ Sigmoid Output

Embedding transforms words into trainable vector representations.

LSTM processes sequential information and captures contextual relationships between words.

## Saved Model

```python
model.save("imdb_lstm_sentiment_model.keras")
```

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Scikit-Learn
