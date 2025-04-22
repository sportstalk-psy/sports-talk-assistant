
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("psychologists_base.json", "r", encoding="utf-8") as f:
    psychologists = json.load(f)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def find_relevant_psychologists(query, top_n=2, threshold=0.35):
    query_embedding = np.array(get_embedding(query)).reshape(1, -1)
    results = []
    for person in psychologists:
        desc_embedding = np.array(get_embedding(person["description"])).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, desc_embedding)[0][0]
        results.append((person, similarity))
    relevant = sorted([r for r in results if r[1] >= threshold], key=lambda x: -x[1])
    return [r[0] for r in relevant[:top_n]]

general_phrases = [
    "подбери психолога", "посоветуй психолога", "нужен психолог", 
    "рекомендовать специалиста", "подскажите специалиста",
    "мне нужен специалист", "ищу психолога", "психолог нужен", "психолога для ребенка"
]

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message_raw = request.json.get("message", "")
        user_message = user_message_raw.lower()
        if not user_message:
            return jsonify({"reply": "Пожалуйста, напишите сообщение.", "cards": []})

        is_general = any(phrase in user_message for phrase in general_phrases)

        system_prompt = (
            "Ты — ИИ-ассистент платформы Sports Talk. "
            "Ты помогаешь по вопросам спортивной психологии, ментальной подготовки, поддержке родителей, "
            "отношениям с тренером. Не выдумывай имён, не упоминай другие платформы. "
            "Если запрос слишком общий, уточни, с чем именно хочет обратиться пользователь и не рекомендуй психологов сразу. "
            "Общайся как психолог со стажем работы не менее 10 лет. "
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_message_raw})

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        base_reply = completion.choices[0].message.content

        if is_general:
            return jsonify({"reply": base_reply, "cards": []})

        matches = find_relevant_psychologists(user_message)
        cards = []
        for match in matches:
            cards.append({
                "name": match["name"],
                "description": match["description"],
                "link": match["link"]
            })

        return jsonify({"reply": base_reply, "cards": cards})
    except Exception as e:
        return jsonify({"reply": "Ошибка на сервере. Попробуйте позже.", "cards": []}), 500

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
