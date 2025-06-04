#Emotion Classifier

A lightweight and effective machine learning app that reads a piece of text and predicts the emotion behind it. Whether it's sadness, joy, love, anger, fear, or surprise — this model helps identify emotional tone in real-time.

## Dataset
Dataset url : https://www.kaggle.com/datasets/adhamelkomy/twitter-emotion-dataset
The model is trained on the Twitter Emotion Dataset, which contains:
- 416,809 Twitter messages
- 6 emotion categories:
  - Sadness (0)
  - Joy (1)
  - Love (2)
  - Anger (3)
  - Fear (4)
  - Surprise (5)

## Approach

### Text Preprocessing
- Convert text to lowercase
- Remove URLs, special characters, and numbers
- Remove stopwords
- Tokenize text

### Feature Extraction
- TF-IDF vectorization with:
  - 5,000 maximum features
  - Unigrams and bigrams
  - Sublinear TF scaling
  - L2 normalization

### Model
- Logistic Regression with:
  - Multinomial classification
  - Class weights for handling imbalance
  - SAGA optimizer
  - L2 regularization

### Evaluation
- 5-fold cross-validation
- Confusion matrix
- Classification report
- Accuracy score

### Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the classifier:
```bash
streamlit run emotion_classifier.py
```

The model will:
- Load existing model if available
- Train new model if needed
- Provide interactive text classification
- Show top 3 predictions with confidence scores