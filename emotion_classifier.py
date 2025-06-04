# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import streamlit as st
# import joblib
# import re
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk
# from scipy import sparse

# # Only download if not already present
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
    
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

# # Emotion mapping
# emotion_map = {
#     0: 'sadness',
#     1: 'joy',
#     2: 'love',
#     3: 'anger',
#     4: 'fear',
#     5: 'surprise'
# }

# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
    
#     # Remove URLs
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
#     # Remove special characters and numbers
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\d+', '', text)
    
#     # Remove extra whitespace
#     text = ' '.join(text.split())
    
#     # Tokenize and remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = word_tokenize(text)
#     tokens = [t for t in tokens if t not in stop_words]
    
#     return ' '.join(tokens)

# def load_and_preprocess_data():
#     # Load the dataset
#     df = pd.read_csv('text.csv')
    
#     # Preprocess text
#     df['processed_text'] = df['text'].apply(preprocess_text)
    
#     X = df['processed_text']
#     y = df['label']
    
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     # Create TF-IDF vectorizer with improved parameters
#     vectorizer = TfidfVectorizer(
#         max_features=5000,        # Reduced features to save memory
#         min_df=5,                 # Increased minimum document frequency
#         max_df=0.9,              # Adjusted maximum document frequency
#         ngram_range=(1, 2),      # Use both unigrams and bigrams
#         sublinear_tf=True,       # Apply sublinear scaling
#         norm='l2'                # L2 normalization
#     )
    
#     X_train_tfidf = vectorizer.fit_transform(X_train)
#     X_test_tfidf = vectorizer.transform(X_test)
    
#     return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# def train_model(X_train, y_train):
#     # Calculate class weights
#     class_weights = dict(zip(
#         np.unique(y_train),
#         len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
#     ))
    
#     # Train a logistic regression model with improved parameters
#     model = LogisticRegression(
#         max_iter=1000,           # Reduced max iterations
#         multi_class='multinomial',
#         class_weight=class_weights,
#         C=0.1,                   # Reduced C for stronger regularization
#         solver='saga',           # Using saga solver
#         tol=1e-4,               # Adjusted tolerance
#         random_state=42,
#         n_jobs=-1               # Use all available cores
#     )
    
#     # Perform cross-validation
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
#     st.write(f'Cross-validation scores: {cv_scores}')
#     st.write(f'Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')
    
#     # Train the final model
#     model.fit(X_train, y_train)
#     return model

# def evaluate_model(model, X_test, y_test):
#     # Make predictions
#     y_pred = model.predict(X_test)
    
#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, y_pred)
    
#     # Generate confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
    
#     # Generate classification report
#     report = classification_report(y_test, y_pred, target_names=[emotion_map[i] for i in range(6)])
    
#     return accuracy, cm, report

# def plot_confusion_matrix(cm):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=[emotion_map[i] for i in range(6)],
#                 yticklabels=[emotion_map[i] for i in range(6)])
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     return plt

# def main():
#     st.title('Emotion Classifier')
    
#     # Try to load existing model and vectorizer
#     try:
#         model = joblib.load('emotion_model.joblib')
#         vectorizer = joblib.load('vectorizer.joblib')
#         st.success('Loaded existing model successfully!')
        
#         # Load a small portion of data for evaluation
#         df = pd.read_csv('text.csv')
#         X_test = df['text'].sample(n=1000, random_state=42)
#         y_test = df['label'].sample(n=1000, random_state=42)
        
#         # Preprocess and transform test data
#         X_test_processed = X_test.apply(preprocess_text)
#         X_test_tfidf = vectorizer.transform(X_test_processed)
        
#         # Evaluate model
#         accuracy, cm, report = evaluate_model(model, X_test_tfidf, y_test)
        
#     except (FileNotFoundError, Exception) as e:
#         st.warning('No existing model found or error loading model. Training new model...')
        
#         # Load and preprocess data
#         with st.spinner('Loading and preprocessing data...'):
#             X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
        
#         # Train model
#         with st.spinner('Training model...'):
#             model = train_model(X_train, y_train)
        
#         # Evaluate model
#         accuracy, cm, report = evaluate_model(model, X_test, y_test)
        
#         # Save model and vectorizer
#         joblib.dump(model, 'emotion_model.joblib')
#         joblib.dump(vectorizer, 'vectorizer.joblib')
#         st.success('Model trained and saved successfully!')
    
#     # Display results
#     st.write(f'Model Accuracy: {accuracy:.2f}')
    
#     # Display confusion matrix
#     st.write('Confusion Matrix:')
#     fig = plot_confusion_matrix(cm)
#     st.pyplot(fig)
    
#     # Display classification report
#     st.write('Classification Report:')
#     st.text(report)
    
#     # Create input interface
#     st.write('---')
#     st.write('Try it yourself!')
#     user_input = st.text_area('Enter text to classify:', '')
    
#     if user_input:
#         # Preprocess input text
#         processed_input = preprocess_text(user_input)
        
#         # Transform input
#         input_vector = vectorizer.transform([processed_input])
        
#         # Make prediction
#         prediction = model.predict(input_vector)[0]
#         probabilities = model.predict_proba(input_vector)[0]
        
#         # Get top 3 predictions
#         top_3_idx = np.argsort(probabilities)[-3:][::-1]
        
#         st.write('Top 3 Predictions:')
#         for idx in top_3_idx:
#             st.write(f'{emotion_map[idx]}: {probabilities[idx]:.2f}')
        
#         # Display the top prediction with a colored box
#         top_emotion = emotion_map[top_3_idx[0]]
#         top_prob = probabilities[top_3_idx[0]]
        
#         st.write('---')
#         st.write('Top Prediction:')
#         st.markdown(f"""
#         <div style='background-color: #000000; padding: 20px; border-radius: 5px;'>
#             <h3 style='margin: 0;'>{top_emotion.upper()}</h3>
#             <p style='margin: 5px 0 0 0;'>Confidence: {top_prob:.2f}</p>
#         </div>
#         """, unsafe_allow_html=True)

# if __name__ == '__main__':
#     main() 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from scipy import sparse

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def load_and_preprocess_data():
    df = pd.read_csv('text.csv')
    df['processed_text'] = df['text'].apply(preprocess_text)
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm='l2'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

def train_model(X_train, y_train):
    class_weights = dict(zip(
        np.unique(y_train),
        len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
    ))
    model = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        class_weight=class_weights,
        C=0.1,
        solver='saga',
        tol=1e-4,
        random_state=42,
        n_jobs=-1
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
    st.write(f'Cross-validation scores: {cv_scores}')
    st.write(f'Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[emotion_map[i] for i in range(6)])
    return accuracy, cm, report

def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[emotion_map[i] for i in range(6)],
                yticklabels=[emotion_map[i] for i in range(6)])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def main():
    st.title('Emotion Classifier')
    try:
        model = joblib.load('emotion_model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        st.success('Loaded existing model successfully!')
        df = pd.read_csv('text.csv')
        X_test = df['text'].sample(n=1000, random_state=42)
        y_test = df['label'].sample(n=1000, random_state=42)
        X_test_processed = X_test.apply(preprocess_text)
        X_test_tfidf = vectorizer.transform(X_test_processed)
        accuracy, cm, report = evaluate_model(model, X_test_tfidf, y_test)
    except (FileNotFoundError, Exception) as e:
        st.warning('No existing model found or error loading model. Training new model...')
        with st.spinner('Loading and preprocessing data...'):
            X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()
        with st.spinner('Training model...'):
            model = train_model(X_train, y_train)
        accuracy, cm, report = evaluate_model(model, X_test, y_test)
        joblib.dump(model, 'emotion_model.joblib')
        joblib.dump(vectorizer, 'vectorizer.joblib')
        st.success('Model trained and saved successfully!')
    st.write(f'Model Accuracy: {accuracy:.2f}')
    st.write('Confusion Matrix:')
    fig = plot_confusion_matrix(cm)
    st.pyplot(fig)
    st.write('Classification Report:')
    st.text(report)
    st.write('---')
    st.write('Try it yourself!')
    user_input = st.text_area('Enter text to classify:', '')
    if user_input:
        processed_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([processed_input])
        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        st.write('Top 3 Predictions:')
        for idx in top_3_idx:
            st.write(f'{emotion_map[idx]}: {probabilities[idx]:.2f}')
        top_emotion = emotion_map[top_3_idx[0]]
        top_prob = probabilities[top_3_idx[0]]
        st.write('---')
        st.write('Top Prediction:')
        st.markdown(f"""
        <div style='background-color: #000000; padding: 20px; border-radius: 5px;'>
            <h3 style='margin: 0;'>{top_emotion.upper()}</h3>
            <p style='margin: 5px 0 0 0;'>Confidence: {top_prob:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
