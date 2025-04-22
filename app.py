
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
            return jsonify({"response": "Пожалуйста, напишите сообщение."})

        is_general = any(phrase in user_message for phrase in general_phrases)

        system_prompt = (
            "Ты — ИИ-ассистент платформы Sports Talk. "
            "Ты помогаешь по вопросам спортивной психологии, ментальной подготовки, поддержке родителей, "
            "отношениям с тренером. Не выдумывай имён, не упоминай другие платформы. "
            "Если запрос слишком общий (например, просто 'посоветуй психолога'), уточни, с чем именно хочет обратиться пользователь и не рекомендуй психологов сразу, только если есть уточнение в текущем вопросе или было ранее сегодня. "
            "Не предлагай специалистов без запроса. "
            "Если даёшь советы или упражнения — можешь использовать HTML-форматирование: <strong> для заголовков, <ul> и <li> для списков, <br> для переносов строк."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_message_raw})

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        base_reply = completion.choices[0].message.content

        if is_general:
            return jsonify({"response": base_reply})

        matches = find_relevant_psychologists(user_message)
        if matches:
            base_reply += "\n\nПо вашему запросу могу порекомендовать следующих специалистов:"
            for match in matches:
                base_reply += (
                    f"\n👤 <strong>{match['name']}</strong><br>"
                    f"{match['description']}<br>"
                    f"<a href='{match['link']}' target='_blank'>Посмотреть профиль психолога</a>"
                )

        return jsonify({"response": base_reply})

    except Exception as e:
        return jsonify({"response": "Ошибка на сервере. Попробуйте позже."}), 500

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
