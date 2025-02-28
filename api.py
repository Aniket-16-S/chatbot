from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

intents = [
    {"tag": "greeting",
     "patterns": ["Hi", "Hello", "Hey", "namaskar"],
     "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]},
    { "tag" : "ask",
      "patterns" : ["How are you ?", "What's up", "howdy?"],
      "responses" : ["Im doing fine .. how about you ?"]
    },
    
    {"tag": "goodbye",
     "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
     "responses": ["Goodbye", "See you later", "Take care"]},
    
    {"tag": "thanks",
     "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
     "responses": ["You're welcome", "No problem", "Glad I could help"]},
    
    {"tag": "about",
     "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
     "responses": ["I am an FAQ chatbot designed to assist you.", "My purpose is to answer your queries.", "I provide automated responses to common questions."]},

    {"tag": "help",
     "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
     "responses": ["Sure, what do you need help with?", "I'm here to assist. Please ask your question.", "Let me know how I can help you."]},

    {"tag": "professor_info",
     "patterns": ["Who is the professor", "Tell me about my professor", "Who teaches this course"],
     "responses": ["Your professor is Dr. XYZ, and they specialize in Machine Learning.", "Dr. XYZ is your professor for this subject.", "Professor XYZ teaches this course."]},

    {"tag": "college_info",
     "patterns": ["Where is the college", "Tell me about the college", "What is the college name"],
     "responses": ["Your college is Finolex Academy of Management and Technology.", "The college is located in Ratnagiri.", "Finolex Academy is known for its engineering programs."]},

    {"tag": "subject_info",
     "patterns": ["What subjects do we have", "Tell me about my subjects", "List my courses"],
     "responses": ["You have subjects like Data Structures, Machine Learning, and Computer Networks.", "Your subjects include Programming, Database Management, and AI.", "The courses you are studying are tailored for engineering students."]},

    {"tag": "exam_info",
     "patterns": ["When are the exams", "Exam dates", "Tell me the exam schedule"],
     "responses": ["The exams will be held in March. Please check the official notice.", "Exam schedules are available on the college portal.", "Your exams are expected to start next month."]},

    {"tag": "assignment_info",
     "patterns": ["When is the assignment due", "Tell me about my assignments", "What assignments do I have"],
     "responses": ["Your assignments are due by next Monday.", "The deadline for assignments is this Friday.", "You have an assignment on Data Science due next week."]},

    {"tag": "sports_event",
     "patterns": ["When is the next sports event", "Tell me about sports events", "Is there any sports competition"],
     "responses": ["The next sports event is scheduled for next month.", "You can check the sports event schedule on the notice board.", "A football tournament is happening this weekend."]},

    {"tag": "fest_info",
     "patterns": ["When is the college fest", "Tell me about the annual fest", "What events are in the fest"],
     "responses": ["The annual fest is in December with various cultural and technical events.", "Your college fest includes hackathons, music competitions, and dance events.", "The fest will have fun activities, competitions, and guest lectures."]},

    {"tag": "library_info",
     "patterns": ["What are library timings", "When is the library open", "Tell me about the library"],
     "responses": ["The library is open from 9 AM to 8 PM.", "You can visit the library during college hours.", "The library has a great collection of books and is open for students all day."]},

    {"tag": "canteen_info",
     "patterns": ["Is the canteen open", "What food is available in the canteen", "Tell me about the canteen"],
     "responses": ["The canteen serves fresh food from 8 AM to 6 PM.", "You can get snacks, meals, and beverages in the canteen.", "The canteen menu includes vegetarian and fast-food options."]},

    {"tag": "contact_info",
     "patterns": ["How do I contact administration", "Where is the admin office", "Tell me the contact details"],
     "responses": ["You can contact administration at admin@college.edu.", "The administration office is on the first floor of Block A.", "For any official queries, visit the admin office during working hours."]}
]

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot_response(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand."

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data.get("message", "")
    response = chatbot_response(user_message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
