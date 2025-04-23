
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

# Хранилище сообщений и состояния рекомендаций по пользователям (по IP)
message_history = defaultdict(list)
recommendation_state = defaultdict(lambda: {"since_last": 100, "last_asked_general": False})

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("psychologists_base.json", "r", encoding="utf-8") as f:
    psychologists = json.load(f)
print(f"🔍 Загрузка: найдено {len(psychologists)} психологов.")

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
        print(f"🔗 Сходство с {person['name']}: {similarity:.3f}")
        results.append((person, similarity))
    relevant = sorted([r for r in results if r[1] >= threshold], key=lambda x: -x[1])
    return [r[0] for r in relevant[:top_n]]

# Общие фразы без конкретного запроса
general_phrases = [
    "подбери психолога", "посоветуй психолога", "нужен психолог", 
    "рекомендовать специалиста", "подскажите специалиста",
    "мне нужен специалист", "ищу психолога", "психолог нужен", "психолога для ребенка"
]

# Фразы, при которых пользователь сам просит менеджера
manager_phrases = [
    "свяжите с менеджером", "связаться с менеджером", "где менеджер", 
    "не могу записаться", "помогите записаться", "саппорт", "поддержка"
]

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message_raw = request.json.get("message", "")
        user_message = user_message_raw.lower()
        if not user_message:
            return jsonify({"response": "Пожалуйста, напишите сообщение."})

        user_ip = request.remote_addr
        message_history[user_ip].append(user_message_raw)
        if len(message_history[user_ip]) > 10:
            message_history[user_ip].pop(0)

        state = recommendation_state[user_ip]

        is_general = any(phrase in user_message for phrase in general_phrases)
        wants_manager = any(phrase in user_message for phrase in manager_phrases)
        has_detail = not is_general  # Простейшая проверка: если не общее — значит уточнение есть

        system_prompt = (
            "Ты — ИИ-ассистент платформы Sports Talk. "
            "Ты помогаешь по вопросам спортивной психологии, ментальной подготовки, поддержке родителей, "
            "отношениям с тренером. Не выдумывай имён, не упоминай другие платформы. "
            "Если запрос слишком общий (например, просто 'посоветуй психолога'), уточни, с чем именно хочет обратиться пользователь. "
            "Не предлагай специалистов без запроса. Общайся как психолог со стажем работы не менее 10 лет. "
            "Ты не психолог, но хорошо разбираешься в психологии и можешь отвечать на простые вопросы и предлагать упражнения из научной и другой литературы."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_message_raw})

        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages
        )

        base_reply = completion.choices[0].message.content

        if is_general and not has_detail:
            state["last_asked_general"] = True
            state["since_last"] += 1
            return jsonify({"response": base_reply + "\n\nМожете уточнить, с чем именно вы хотите поработать?"})

        if state["last_asked_general"] and has_detail:
            state["last_asked_general"] = False
            state["since_last"] = 0
            matches = find_relevant_psychologists(user_message)
            if matches:
                base_reply += "\n\nПо Вашему запросу могу порекомендовать следующих специалистов:"
                for match in matches:
                    base_reply += (
                        f"<br><br><strong>👤 {match['name']}</strong><br>"
                        f"{match['description']}<br>"
                        f"<a href='{match['link']}' target='_blank'>Посмотреть профиль психолога</a>"
                    )
            return jsonify({"response": base_reply})

        if state["since_last"] < 3:
            state["since_last"] += 1
            return jsonify({"response": base_reply})

        matches = find_relevant_psychologists(user_message)
        if matches:
            state["since_last"] = 0
            base_reply += "\n\nПо Вашему запросу могу порекомендовать следующих специалистов:"
            for match in matches:
                base_reply += (
                    f"<br><br><strong>👤 {match['name']}</strong><br>"
                    f"{match['description']}<br>"
                    f"<a href='{match['link']}' target='_blank'>Посмотреть профиль психолога</a>"
                )
        else:
            state["since_last"] += 1
            if is_general or wants_manager:
                base_reply += (
                    "\n\nВижу, что нам нужно чуть больше данных, чтобы порекомендовать соответствующего специалиста. "
                    "Вы можете связаться с нашим менеджером — мы обязательно поможем вам подобрать психолога и оформить запись:\n\n"
                    "<a href='https://wa.me/+79112598408' target='_blank' style='color:#ebf5ff;'>📲 Связаться с менеджером в WhatsApp</a>"
                )

        return jsonify({"response": base_reply })

    except Exception as e:
        print("Ошибка сервера:", str(e))
        return jsonify({"response": "Ошибка на сервере. Попробуйте позже."}), 500

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
