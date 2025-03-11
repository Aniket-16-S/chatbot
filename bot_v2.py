import os
import json
import random
import nltk
import torch
import streamlit as st
import sqlite3
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet
from textblob import TextBlob
from transformers import pipeline
from flask_cors import CORS

# Download required nltk data
nltk.download('wordnet')
nltk.download('punkt')

# Load pre-trained model for embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load FAQ data from SQLite
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS faqs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT
    )
""")
conn.commit()

# Function to get responses from database
def get_response(user_input):
    cursor.execute("SELECT question, answer FROM faqs")
    faqs = cursor.fetchall()
    questions = [q[0] for q in faqs]
    answers = [q[1] for q in faqs]
    
    embeddings = model.encode(questions)
    input_embedding = model.encode([user_input])
    similarities = torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings), torch.tensor(input_embedding), dim=1
    )
    best_match = torch.argmax(similarities).item()
    
    if similarities[best_match] > 0.7:
        return answers[best_match]
    else:
        return "I'm not sure about that. Can you rephrase or provide more details?"

# Data Augmentation with Synonyms
def augment_text(text):
    words = nltk.word_tokenize(text)
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words.append(synonym)
        else:
            new_words.append(word)
    return " ".join(new_words)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    response = get_response(user_input)
    return jsonify({"response": response})

# Streamlit UI
def main():
    st.title("Advanced AI Chatbot")
    st.write("Chat with our AI-powered assistant.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:")

    if user_input:
        response = get_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Chatbot", response))

    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**{sender}:** {message}")
        else:
            st.markdown(f":speech_balloon: **{sender}:** {message}")

if __name__ == "__main__":
    main()
    app.run(port=5000, debug=True)
