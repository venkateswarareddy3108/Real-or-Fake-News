import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = text.split()
    # Remove stopwords and lemmatize
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    return " ".join(processed_tokens)


# Load the model and vectorizer
with open('xgboost.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# App title
st.title("📰 Fake/Real News Classifier")
st.write("This application classifies news articles as **Real** or **Fake** based on their content.")

# Input area for the news article
news_article = st.text_area("📝 Enter News Article Here:", height=200, placeholder="Paste or type your news article...")

# Classify button
if st.button("🔍 Classify"):
    if news_article.strip():  # Ensure the input is not empty
        # Preprocess the input text
        preprocessed_text = preprocess_text(news_article)

        # Transform the input text using the vectorizer
        transformed_text = vectorizer.transform([preprocessed_text])

        # Predict using the XGBoost model
        prediction = xgb_model.predict(transformed_text)

        # Display result
        result = "✅ Real News" if prediction[0] == 1 else "❌ Fake News"
        st.success(f"The article is classified as: **{result}**")
    else:
        st.error("❗ Please enter a valid news article.")





