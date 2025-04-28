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

# Считаем эмбеддинги всех психологов заранее
ready_embeddings = []
for person in psychologists:
    embedding = None
    try:
        embedding = np.array(
            client.embeddings.create(
                model="text-embedding-3-small",
                input=person["description"]
            ).data[0].embedding
        )
        ready_embeddings.append({"person": person, "embedding": embedding})
        print(f"✅ Эмбеддинг для {person['name']} рассчитан.")
    except Exception as e:
        print(f"⚠️ Ошибка при расчёте эмбеддинга для {person['name']}: {str(e)}")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def find_relevant_psychologists(query, top_n=2, threshold=0.35):
    query_embedding = np.array(
        client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding
    ).reshape(1, -1)

    results = []
    for item in ready_embeddings:
        similarity = cosine_similarity(query_embedding, item["embedding"].reshape(1, -1))[0][0]
        print(f"🔗 Сходство с {item['person']['name']}: {similarity:.3f}")
        results.append((item["person"], similarity))

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
    "Ты помогаешь спортсменам, их родителям и тренерам по вопросам спортивной психологии. "
    "Работаешь только с темами: ментальная подготовка, предстартовое волнение, выгорание, травмы, страх ошибки, отношения с тренером, мотивация, самооценка, родительское давление. "
    "Если запрос не по теме спортивной психологии — мягко сообщи, что можешь помогать только в этой области. "
    "Если в запросе упоминаются слова 'психолог', 'специалист', 'консультация', 'помощь', 'порекомендуй' и есть описание проблемы (например, стресс, волнение, страх, выгорание, мотивация, тревога) — сразу предложи специалистов из базы. "
    "Если это первое сообщение и проблема не указана — сначала уточни запрос у пользователя. "
    "После уточнения запроса задай все необходимые вопросы ДО того, как начнёшь рекомендовать психологов. "
    "Обращай внимание на возраст спортсмена: если пользователь ищет психолога для ребёнка или подростка, не предлагай специалистов, которые работают только со взрослыми. "
    "Все консультации на платформе проходят онлайн, поэтому не спрашивай у пользователя город или страну проживания. "
    "Только после завершения всех уточнений используй фразу 'По Вашему запросу могу порекомендовать следующих специалистов:' "
    "и сразу перечисли специалистов без дополнительных вопросов. "
    "Отвечай кратко, до 500 символов. "
    "Не выдумывай новых психологов и не упоминай сторонние платформы. "
    "Общайся дружелюбно и профессионально, как психолог со стажем более 10 лет."
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
