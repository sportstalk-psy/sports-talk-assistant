
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

general_phrases = [
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

        if found_age and state.get("last_problem_message"):
            user_message_raw = state["last_problem_message"]
            user_message = user_message_raw.lower()

        is_general = any(phrase in user_message for phrase in general_phrases)
        wants_manager = any(phrase in user_message for phrase in manager_phrases)
        has_detail = any(word in user_message for word in valid_problem_keywords)

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

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message_raw}]
        completion = client.chat.completions.create(model="gpt-4-turbo", messages=messages)
        base_reply = completion.choices[0].message.content

        if wants_manager:
            base_reply += "\n\nЕсли у вас возникли трудности — "
            base_reply += "<br><a href='https://wa.me/+79112598408' target='_blank'>📲 Связаться с менеджером</a>"
            return jsonify({"response": base_reply})

        if is_general and not has_detail:
            state["last_asked_general"] = True
            clarify_text = random.choice(templates["clarify_problem"])
            return jsonify({"response": base_reply + "\n\n" + clarify_text})

        if has_detail:
            state["problem_collected"] = True
            state["last_problem_message"] = user_message_raw

        if not state["problem_collected"]:
            return jsonify({"response": random.choice(templates["clarify_problem"])})

        if not state["age_collected"]:
            return jsonify({"response": random.choice(templates["request_age"])})

        matches = find_relevant_psychologists(user_message, user_age_group=state["user_age_group"])
        if matches:
            start_rec_text = random.choice(templates["start_recommendation"])
            base_reply += "\n\n" + start_rec_text
            for match in matches:
                base_reply += f"<br><br><strong>👤 {match['name']}</strong><br>{match['description']}<br><a href='{match['link']}' target='_blank'>Посмотреть профиль</a>"
        else:
            base_reply += "\n\nНе удалось найти подходящего специалиста. Попробуйте уточнить запрос."

        return jsonify({"response": base_reply})

    except Exception as e:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🚨 [{now}] Ошибка сервера: {str(e)}")
        return jsonify({"response": "Произошла ошибка на сервере. Попробуйте позже."}), 500

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
