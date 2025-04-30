import os
import re
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Константы минимального и максимального возраста клиентов
MIN_CLIENT_AGE = 5
MAX_CLIENT_AGE = 80

# Загрузка переменных окружения
load_dotenv()

# Настройка Flask-приложения
app = Flask(__name__)
CORS(app)

# Инициализация OpenAI-клиента
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Загрузка базы психологов
with open("psychologists_base.json", "r", encoding="utf-8") as f:
    psychologists = json.load(f)


def get_embedding(text: str) -> np.ndarray:
    """
    Получение эмбеддинга текста с помощью OpenAI API.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)


def classify_age_group(message: str) -> str:
    """
    Определяет, является ли пользователь ребёнком (<18) или взрослым (>=18).
    Возвращает 'child' или 'adult'.
    """
    prompt_system = (
        "Ты — помощник, который определяет возрастную группу клиента. "
        f"Если клиент младше 18 лет, ответь 'child', иначе 'adult'. "
        f"Работать можно только с возрастами от {MIN_CLIENT_AGE} до {MAX_CLIENT_AGE}."
    )
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": message}
        ]
    )
    grp = resp.choices[0].message.content.strip().lower()
    return 'child' if 'child' in grp else 'adult'


def find_relevant_psychologists(query: str, age_group: str, top_n: int = 2, threshold: float = 0.3) -> list[dict]:
    """
    Поиск подходящих психологов по смысловой близости описания и по возрастной группе.
    - Для детей (child): отбираем психологов, чьи диапазоны пересекаются с [{MIN_CLIENT_AGE}, 17].
    - Для взрослых (adult): отбираем психологов, чьи диапазоны пересекаются с [18, {MAX_CLIENT_AGE}].
    """
    candidates = []
    for p in psychologists:
        p_min = p.get('min_age', MIN_CLIENT_AGE)
        p_max = p.get('max_age', MAX_CLIENT_AGE)
        if age_group == 'child':
            if p_min < 18 and p_max >= MIN_CLIENT_AGE:
                candidates.append(p)
        else:
            if p_min <= MAX_CLIENT_AGE and p_max >= 18:
                candidates.append(p)

    # Эмбеддинг запроса
    query_emb = get_embedding(query).reshape(1, -1)
    scored = []
    for person in candidates:
        try:
            desc_emb = get_embedding(person.get('description', '')).reshape(1, -1)
            sim = cosine_similarity(query_emb, desc_emb)[0][0]
            if sim >= threshold:
                scored.append((person, sim))
        except Exception as e:
            print(f"⚠️ Ошибка эмбеддинга для {person.get('name')}: {e}")

    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:top_n]]


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"response": "Пожалуйста, напишите сообщение."})

        # Системная инструкция: только спортивные темы и рекомендации только из базы
        system_prompt = (
            "Ты — ИИ-ассистент Sports Talk. Отвечай только на вопросы, связанные со спортом. "
            "Если сообщение не про спорт, ответь: 'Извините, я могу отвечать только на темы, связанные со спортом.' "
            "Рекомендовать психологов можно только из базы psychologists_base.json."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Генерируем ответ на спортивную тему
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages
        )
        base_reply = completion.choices[0].message.content.strip()

        # Рекомендация психологов при запросе
        if re.search(r"психолог", user_message, flags=re.IGNORECASE):
            age_group = classify_age_group(user_message)
            matches = find_relevant_psychologists(user_message, age_group)
            if matches:
                base_reply += "\n\nМогу порекомендовать специалистов:"
                for m in matches:
                    base_reply += (
                        f"<br><br><strong>👤 {m.get('name')}</strong><br>"
                        f"{m.get('description', '')}<br>"
                        f"<a href='{m.get('link')}' target='_blank'>Посмотреть профиль</a>"
                    )

        return jsonify({"response": base_reply})

    except Exception as e:
        print("🚨 Ошибка сервера:", str(e))
        return jsonify({"response": "Произошла ошибка на сервере. Попробуйте позже."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 80)))