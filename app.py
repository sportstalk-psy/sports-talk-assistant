
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
import random
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import datetime

load_dotenv()

message_history = defaultdict(list)
recommendation_state = defaultdict(lambda: {
    "since_last": 100,
    "last_asked_general": False,
    "waiting_for_age": False,
    "problem_collected": False,
    "age_collected": False,
    "user_age_group": None,
    "last_problem_message": None
})

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("psychologists_base.json", "r", encoding="utf-8") as f:
    psychologists = json.load(f)

with open("templates.json", "r", encoding="utf-8") as f:
    templates = json.load(f)

ready_embeddings = []
for person in psychologists:
    try:
        embedding = np.array(
            client.embeddings.create(
                model="text-embedding-3-small",
                input=person["description"]
            ).data[0].embedding
        )
        ready_embeddings.append({"person": person, "embedding": embedding})
    except Exception as e:
        print(f"⚠️ Ошибка при расчёте эмбеддинга для {person['name']}: {str(e)}")

def find_relevant_psychologists(query, top_n=2, threshold=0.35, user_age_group=None):
    query_embedding = np.array(
        client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding
    ).reshape(1, -1)

    results = []
    for item in ready_embeddings:
        person = item["person"]
        if "age_group" not in person:
            print(f"⚠️ У психолога {person['name']} не указан age_group.")
        if user_age_group == "children" and person.get("age_group") not in ["children", "all"]:
            print(f"🔽 Пропущен: {person['name']} — возрастная группа не совпала")
            continue
        if user_age_group == "adults" and person.get("age_group") not in ["adults", "all"]:
            print(f"🔽 Пропущен: {person['name']} — возрастная группа не совпала")
            continue
        similarity = cosine_similarity(query_embedding, item["embedding"].reshape(1, -1))[0][0]
        results.append((person, similarity))

    relevant = sorted([r for r in results if r[1] >= threshold], key=lambda x: -x[1])
    return [r[0] for r in relevant[:top_n]]

#general_phrases = [
    "подбери психолога", "посоветуй психолога", "нужен психолог",
    "рекомендовать специалиста", "подскажите специалиста",
    "мне нужен специалист", "ищу психолога", "психолог нужен", "психолога для ребенка"
]

valid_problem_keywords = [
    "страх", "волнение", "мотивация", "выгорание", "травма", "ошибка",
    "отношения", "уверенность", "провал", "самооценка", "кризис",
    "депрессия", "стресс", "эмоции", "переживания"
]

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

        print(f"📩 Сообщение от {user_ip}: {user_message_raw}")

        if not user_message:
            return jsonify({"response": "Пожалуйста, напишите сообщение."})

        user_ip = request.remote_addr
        message_history[user_ip].append(user_message_raw)
        if len(message_history[user_ip]) > 10:
            message_history[user_ip].pop(0)

        state = recommendation_state[user_ip]
        found_age = False

        # Возраст
        age_match = re.search(r"\b(сыну|дочке|мальчику|девочке|спортсмену|ребёнку|ребенку|подростку)?\s*(\d{1,2})", user_message)
        if age_match:
            found_age = True
            try:
                age = int(age_match.group(2))
                if 5 <= age <= 18:
                    state["age_collected"] = True
                    state["user_age_group"] = "children"
                elif 19 <= age <= 80:
                    state["age_collected"] = True
                    state["user_age_group"] = "adults"
            except ValueError:
                pass
        else:
            try:
                age = int(user_message.strip())
                if 5 <= age <= 18:
                    state["age_collected"] = True
                    state["user_age_group"] = "children"
                elif 19 <= age <= 80:
                    state["age_collected"] = True
                    state["user_age_group"] = "adults"
            except ValueError:
                pass

        # Если сообщение содержит только возраст — подставим предыдущее сообщение
        if found_age and state.get("last_problem_message") and len(user_message.split()) <= 3:
            print("🔄 Использую последнее сообщение с проблемой.")
            user_message_raw = state["last_problem_message"]
            user_message = user_message_raw.lower()

        #is_general = any(phrase in user_message for phrase in general_phrases)
        wants_manager = any(phrase in user_message for phrase in manager_phrases)
        
# пока отрубаем ручную проверку запроса has_detail = any(word in user_message for word in valid_problem_keywords)

        # --- Новый system_prompt ---
        system_prompt = (
            "Ты — ИИ-ассистент платформы спортивной психологии Sports Talk, тебя зовут Марк. "
            "Ты помогаешь спортсменам, родителям и тренерам по темам ментальной подготовки. "
            "Сфера твоей работы: предстартовое волнение, выгорание, травмы, страх ошибки, мотивация, самооценка, конфликты, давление, стресс и другие психоэмоциональные состояния, возникающие в спорте. "
            "Если пользователь просит, ты можешь порекомендовать психологов — но никогда не придумывай имена и описания. Карточки специалистов формируются отдельно и подставляются платформой. "
            "Если пользователь описал проблему, требующую личной работы, скажи: 'По вашему запросу могу порекомендовать специалистов'. "
            "Ты также можешь делиться техниками: дыхательные упражнения, психотехнические игры, приёмы саморегуляции, упражнения из спортивной психологии. "
            "Тебе можно общаться на околотематические темы — фильмы, книги, вдохновляющие примеры, ментальные ритуалы и т.д., если это поддерживает мотивацию. "
            "Если вопрос не по теме спорта — мягко объясни, что ты работаешь только со спортивной психологией. "
            "Отвечай дружелюбно и профессионально, кратко и понятно. Не повторяй одно и то же. Не запрашивай возраст, если он уже указан. "
            "Все консультации проходят онлайн на платформе Sports Talk."
)

        # Собираем последние сообщения пользователя
        history = message_history[user_ip][-5:]  # последние 5 реплик максимум

        # Формируем историю диалога для ИИ
        messages = [{"role": "system", "content": system_prompt}]
        for user_msg in history:
            messages.append({"role": "user", "content": user_msg})

        completion = client.chat.completions.create(model="gpt-4-turbo", messages=messages)
        base_reply = completion.choices[0].message.content

        # --- Новый способ: запрашиваем возраст и запрос только при прямом запросе психолога ---
        is_direct_request = any(kw in user_message for kw in ["подбери", "порекомендуй", "хочу записаться", "ищу психолога", "нужен специалист", "посоветуй психолога"])

        # --- Проверка: понял ли ИИ проблему сам ---
        if any(keyword in base_reply.lower() for keyword in ["уточните", "поясните", "расскажите подробнее", "что именно", "с чем связано"]):
            state["problem_collected"] = False
        else:
            state["problem_collected"] = True
            state["last_problem_message"] = user_message_raw

        if wants_manager:
            base_reply += "\n\nЕсли у вас возникли трудности — "
            base_reply += "<br><a href='https://wa.me/+79112598408' target='_blank'>📲 Связаться с менеджером</a>"
            return jsonify({"response": base_reply})

        # --- Если человек просит подобрать психолога, но возраст или проблема не указаны — просим их вместе ---
        if is_direct_request and not (state["problem_collected"] and state["age_collected"]):
            return jsonify({"response": "Напишите, пожалуйста, с какой темой вы хотели бы поработать и возраст человека, для которого нужна консультация."})

        # --- Если и возраст, и проблема есть — подбор психолога (по прямому запросу) ---
        if is_direct_request and state["problem_collected"] and state["age_collected"]:
            matches = find_relevant_psychologists(user_message, user_age_group=state["user_age_group"])
            if matches:
                rec_text = random.choice(templates["start_recommendation"])
                base_reply = rec_text
                for match in matches:
                    base_reply += f"<br><br><strong>👤 {match['name']}</strong><br>{match['description']}<br><a href='{match['link']}' target='_blank'>Посмотреть профиль</a>"
            else:
                base_reply = "К сожалению, не удалось найти подходящего специалиста. Возможно, я еще не так хорошо знаю базу, хотите связаться с менеджером? Или попробуйте немного переформулировать запрос."
            return jsonify({"response": base_reply})

        # --- Если человек просто говорит про тему (например, стресс) — короткий ответ + 1 специалист ---
        if state["problem_collected"] and state["age_collected"]:
            matches = find_relevant_psychologists(user_message, user_age_group=state["user_age_group"])
            if matches:
                base_reply += "<br><br>Если захотите обсудить это подробнее, могу порекомендовать специалиста:"
                match = matches[0]
                base_reply += f"<br><br><strong>👤 {match['name']}</strong><br>{match['description']}<br><a href='{match['link']}' target='_blank'>Посмотреть профиль</a>"

        return jsonify({"response": base_reply})

    except Exception as e:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🚨 [{now}] Ошибка сервера: {str(e)}")
        return jsonify({"response": "Произошла ошибка на сервере. Попробуйте позже."}), 500

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
