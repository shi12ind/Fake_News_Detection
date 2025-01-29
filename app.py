 
import streamlit as st
import pickle
import gzip

# Load Fake News Model
with gzip.open('fake_news_model.pkl.gz', 'rb') as compressed_file:
    fake_news_model = pickle.load(compressed_file)

# Load Random Forest Model
with open('random_forest_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Load TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("📰 Fake News Detector")
st.subheader("Enter a news article below to check if it's real or fake.")

# User Input
user_input = st.text_area("Paste the news article here...", height=200)

if st.button("Check News"):
    if user_input.strip():
        # Convert text to TF-IDF features
        transformed_input = tfidf_vectorizer.transform([user_input])

        # Make Predictions
        fake_news_prediction = fake_news_model.predict(transformed_input)[0]
        rf_prediction = rf_model.predict(transformed_input)[0]

        # Display results
        st.subheader("Prediction Results:")
        st.write(f"📰 **Fake News Model Prediction:** {'Fake' if fake_news_prediction == 1 else 'Real'}")
        st.write(f"🌲 **Random Forest Model Prediction:** {'Fake' if rf_prediction == 1 else 'Real'}")

    else:
        st.warning("⚠️ Please enter a news article before checking.")

st.markdown("### 📌 Steps to Use:")
st.write("""
1️⃣ Copy and paste a news article into the text box.  
2️⃣ Click the "Check News" button.  
3️⃣ The app will analyze the article using two models and display if the news is **Fake** or **Real**.
""")
