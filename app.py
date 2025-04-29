from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import random
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

# Хранилище сообщений и состояния рекомендаций по пользователям (по IP)
message_history = defaultdict(list)
recommendation_state = defaultdict(lambda: {"since_last": 100, "last_asked_general": False, "waiting_for_age": False, "problem_collected": False})

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("psychologists_base.json", "r", encoding="utf-8") as f:
    psychologists = json.load(f)
print(f"🔍 Загрузка: найдено {len(psychologists)} психологов.")

with open("templates.json", "r", encoding="utf-8") as f:
    templates = json.load(f)

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

# Фразы, при которых пользователь сам просит менеджера или сообщает о проблемах с платформой
manager_phrases = [
    "свяжите с менеджером", "связаться с менеджером", "где менеджер", 
    "не могу записаться", "помогите записаться", "саппорт", "поддержка",
    "проблема с регистрацией", "не могу зарегистрироваться", "ошибка при регистрации",
    "проблема с оплатой", "не проходит оплата", "ошибка оплаты",
    "проблема с подключением", "не получается подключиться", "не работает подключение"
]

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message_raw = request.json.get("message", "")
        user_message = user_message_raw.lower()

        # Список фраз, указывающих на непонимание пользователя
        confusion_phrases = ["не поняла", "не понимаю", "что?", "не совсем ясно", "неясно", "не понятно", "не ясно"]


        if not user_message:
            return jsonify({"response": "Пожалуйста, напишите сообщение."})

        user_ip = request.remote_addr
        message_history[user_ip].append(user_message_raw)
        if len(message_history[user_ip]) > 10:
            message_history[user_ip].pop(0)

        state = recommendation_state[user_ip]

        if any(phrase in user_message for phrase in confusion_phrases):
            if state.get("last_problem_message"):
                print("🔄 Восстанавливаю последний запрос пользователя из памяти.")
                user_message_raw = state["last_problem_message"]
                user_message = user_message_raw.lower()
            else:
                print("⚠️ Нет сохранённой последней проблемы. Продолжаем обычную обработку.")


        # Проверка возраста по ключевым словам и числу
        age_keywords = ["лет", "год", "года", "подросток", "ребёнок", "ребенок"]
        if any(word in user_message for word in age_keywords):
            state["age_collected"] = True
        else:
            try:
                age = int(user_message.strip())
                if 5 <= age <= 80:
                    state["age_collected"] = True
            except ValueError:
                pass

        is_general = any(phrase in user_message for phrase in general_phrases)
        wants_manager = any(phrase in user_message for phrase in manager_phrases)
        has_detail = not any(phrase in user_message for phrase in general_phrases)

        # Генерация ответа от GPT
        system_prompt = (
            "Ты — ИИ-ассистент платформы Sports Talk. "
            "Никогда не придумывай имена или описания специалистов. Не предлагай вымышленных психологов. "
            "Все рекомендации специалистов формируются отдельно платформой Sports Talk на основе внутренней базы данных. "
            "Ты помогаешь спортсменам, их родителям и тренерам по вопросам спортивной психологии. "
            "Работаешь только с темами: ментальная подготовка, предстартовое волнение, выгорание, травмы, страх ошибки, отношения с тренером, мотивация, самооценка, родительское давление. "
            "Если запрос не по теме спортивной психологии — мягко сообщи, что можешь помогать только в этой области. "
            "Если упоминаются слова 'психолог', 'специалист', 'консультация', 'помощь', 'порекомендуй' и есть описание проблемы — сразу предложи специалистов из базы. "
            "Сначала уточняй запрос, если проблема не указана. "
            "Обращай внимание на возраст: если ребёнок или подросток — не предлагай специалистов для взрослых. "
            "Все консультации проходят онлайн. "
            "После уточнения используй фразу 'По Вашему запросу могу порекомендовать следующих специалистов:' "
            "Отвечай кратко, до 500 символов. Общайся дружелюбно и профессионально."
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_message_raw})

        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages
        )
        base_reply = completion.choices[0].message.content

        # --- Проверяем: просит ли человек менеджера ---
        if wants_manager:
            base_reply += (
                "\n\nЕсли у вас возникли трудности с регистрацией, оплатой или подключением, "
                "наш менеджер поможет вам решить эти вопросы:"
                "<br><br><a href='https://wa.me/+79112598408' target='_blank' style='color:#ebf5ff;'>📲 Связаться с менеджером</a>"
            )
            return jsonify({"response": base_reply})

        # Проверка на общий запрос без уточнения
        if is_general and not has_detail:
            state["last_asked_general"] = True
            state["since_last"] += 1
            if any(phrase in base_reply.lower() for phrase in ["уточните", "поделитесь", "что именно", "какие проблемы", "с чем именно"]):
                return jsonify({"response": base_reply})
            else:
                clarify_text = random.choice(templates["clarify_problem"])
                return jsonify({"response": base_reply + "\n\n" + clarify_text})

        # Проверка, что уточнение получено
        if state["last_asked_general"] and has_detail:
            state["last_asked_general"] = False
            state["since_last"] = 0
            state["problem_collected"] = True
            return jsonify({"response": base_reply})

        # Переходим к подбору специалистов
        matches = find_relevant_psychologists(user_message)

        if matches:
            start_rec_text = random.choice(templates["start_recommendation"])
            base_reply += "\n\n" + start_rec_text
            for match in matches:
                base_reply += (
                    f"<br><br><strong>👤 {match['name']}</strong><br>"
                    f"{match['description']}<br>"
                    f"<a href='{match['link']}' target='_blank'>Посмотреть профиль психолога</a>"
                )
        else:
            print("👀 Нет подходящих специалистов — отправляем обычный ответ.")

        # ✅ Всегда отправляем базовый ответ!
        return jsonify({"response": base_reply})

    except Exception as e:
        print("Ошибка сервера:", str(e))
        return jsonify({"response": "Ошибка на сервере. Попробуйте позже."}), 500

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
