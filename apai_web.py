from flask import Flask, render_template, request, jsonify
from apai_demo.main import query_llm, PROMPT_FULL

app = Flask(__name__)
history = []

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("user_query", "")
    response = query_llm(PROMPT_FULL, user_query)
    # If LLM error detected, show friendly message
    if response.startswith("[ERROR]") or "api limit" in response.lower() or "failed" in response.lower():
        response = "I am tired Boss. Please try again later."
    history.append({"question": user_query, "response": response})
    return jsonify({
        "response": response,
        "history": history
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
