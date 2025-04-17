
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

# Новый клиент OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"response": "Пожалуйста, напишите сообщение."})

        # Новый формат вызова chat completion
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )
        reply = completion.choices[0].message.content
        return jsonify({"response": reply})

    except Exception as e:
        print("Ошибка сервера:", str(e))
        return jsonify({"response": "Ошибка на сервере. Попробуйте позже."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
