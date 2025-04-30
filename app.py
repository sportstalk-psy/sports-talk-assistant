import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# Загрузка окружения
load_dotenv()

# Настройка Flask
app = Flask(__name__)
CORS(app)

# Инициализация OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Загрузка базы психологов
with open("psychologists_base.json", "r", encoding="utf-8") as f:
    psychologists = json.load(f)

# Формирование системного промпта с прикреплённой базой
system_prompt = (
    "Ты — ИИ-ассистент платформы спортивной психологии Sports Talk, тебя зовут Марк. "
    "Отвечай только на вопросы про спорт и спортивную психологию дружелюбно, ободряющим тоном, "
    "с опорой на мотивацию и будущее. "
    "Если тебя спрашивают предложить психолога — выбирай только из приложенного JSON-файла."
    "Имена психологов обязательно оборачивай так: `<strong style=\"color:#cefa07;\">Имя Фамилия</strong>`. "
    "Для всех остальных жирных выделений (заголовки секций или пункты рекомендаций) используй чистый HTML-тег `<strong>…</strong>`, без внедрённых стилей.\n"    "При выводе ссылки на профиль психолога используй HTML-ссылки с inline-стилем цвета #cefa07:\n"
    "<a href=\"URL_психолога\" target=\"_blank\" style=\"color:#ebf5ff;\">Посмотреть профиль</a>\n"
    "\n\nБаза психологов:\n"
    + json.dumps(psychologists, ensure_ascii=False)
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Пожалуйста, напишите сообщение."})

    # Подготовка сообщений для GPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Вызов модели
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        top_p=1.0,
        messages=messages
    )
    reply = completion.choices[0].message.content.strip()

    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 80)))
