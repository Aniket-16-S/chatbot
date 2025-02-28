document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

function toggleChat() {
    let chatBox = document.getElementById("chat-box");
    chatBox.style.display = chatBox.style.display === "none" ? "block" : "none";
}

function sendMessage() {
    let userInputField = document.getElementById("user-input");
    let userMessage = userInputField.value.trim();
    
    if (userMessage === "") return;

    let chatBox = document.getElementById("chat-box");
    
    // Append user message
    chatBox.innerHTML += `<div><strong>You:</strong> ${userMessage}</div>`;
    userInputField.value = "";
    
    // Send to API
    fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: userMessage })
    })
    .then(response => response.json())
    .then(data => {
        // Append chatbot response
        chatBox.innerHTML += `<div><strong>Bot:</strong> ${data.response}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll
    })
    .catch(error => {
        console.error("Error:", error);
        chatBox.innerHTML += `<div><strong>Bot:</strong> Sorry, there was an error.</div>`;
    });
}
