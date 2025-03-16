import json
import numpy as np
import tensorflow as tf
import random
import nltk
from nltk.stem import LancasterStemmer
from sklearn.preprocessing import LabelEncoder

stemmer = LancasterStemmer()
nltk.download('punkt')

# Load intents from JSON file
with open("faq.json", "r") as file:
    intents = json.load(file)

# Data Preprocessing
words = []
classes = []
documents = []
ignore_words = ["?", "!", ".", ","]

# Tokenize and stem words
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set([stemmer.stem(w.lower()) for w in words if w not in ignore_words]))
classes = sorted(set(classes))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in [stemmer.stem(word.lower()) for word in doc[0]] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Convert to numpy arrays
random.shuffle(training)
train_x, train_y = zip(*training)
train_x = np.array(train_x)
train_y = np.array(train_y)

# Build Model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_x, train_y, epochs=200, batch_size=8, verbose=1)

# Save model
model.save("chatbot_model.h5")

# Load Model
def load_model():
    return tf.keras.models.load_model("chatbot_model.h5")

# Convert user input to a bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return [1 if w in sentence_words else 0 for w in words]

# Classify intent
def classify(sentence):
    model = load_model()
    bow = np.array([clean_up_sentence(sentence)])
    prediction = model.predict(bow)[0]
    ERROR_THRESHOLD = 0.6
    results = [[i, p] for i, p in enumerate(prediction) if p > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

# Generate response
def chatbot_response(text):
    intents_detected = classify(text)
    if intents_detected:
        intent_tag = intents_detected[0][0]
        for intent in intents["intents"]:
            if intent["tag"] == intent_tag:
                return random.choice(intent["responses"])
    return "I'm sorry, I don't understand."

# Start chatbot
print("Chatbot is ready! Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    print("Chatbot:", chatbot_response(user_input))
