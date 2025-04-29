
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

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message_raw}]
        completion = client.chat.completions.create(model="gpt-4-turbo", messages=messages)
        base_reply = completion.choices[0].message.content

        # --- Новый способ определения: нужна ли рекомендация психолога ---
        needs_psychologist = (
            any(kw in user_message for kw in ["психолог", "специалист", "консультация"]) 
            or ("по вашему запросу могу порекомендовать" in base_reply.lower())
            or (state.get("problem_collected", False) and state.get("age_collected", False))
        )

        if needs_psychologist and not state["age_collected"]:
            return jsonify({"response": random.choice(templates["request_age"])})

        # --- НОВАЯ ПРОВЕРКА ОТВЕТА ИИ ---
        if any(keyword in base_reply.lower() for keyword in ["уточните", "поясните", "расскажите подробнее", "что именно", "с чем связано"]):
            # ИИ попросил уточнить проблему
            state["problem_collected"] = False
        else:
            # ИИ понял проблему сам
            state["problem_collected"] = True
            state["last_problem_message"] = user_message_raw

        if wants_manager:
            base_reply += "\n\nЕсли у вас возникли трудности — "
            base_reply += "<br><a href='https://wa.me/+79112598408' target='_blank'>📲 Связаться с менеджером</a>"
            return jsonify({"response": base_reply})

        #if is_general and not state["problem_collected"]:
            #state["last_asked_general"] = True
            #clarify_text = random.choice(templates["clarify_problem"])
            #return jsonify({"response": base_reply + "\n\n" + clarify_text})

        #if not state["problem_collected"]:
            #return jsonify({"response": random.choice(templates["clarify_problem"])})

        if needs_psychologist:
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
    app.run(host="0.0.0.0", port=80)
