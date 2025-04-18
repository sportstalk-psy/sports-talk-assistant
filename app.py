
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Загружаем базу психологов
with open("psychologists_base.json", "r", encoding="utf-8") as f:
    psychologists = json.load(f)

# Получаем эмбеддинг из OpenAI
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Получаем топ-2 наиболее релевантных психолога
def find_relevant_psychologists(query, top_n=2, threshold=0.80):
    query_embedding = np.array(get_embedding(query)).reshape(1, -1)
    results = []

    for person in psychologists:
        desc_embedding = np.array(get_embedding(person["description"])).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, desc_embedding)[0][0]
        results.append((person, similarity))

    # Фильтруем по порогу и сортируем
    relevant = sorted([r for r in results if r[1] >= threshold], key=lambda x: -x[1])
    return [r[0] for r in relevant[:top_n]]

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"response": "Пожалуйста, напишите сообщение."})

        # Генерируем основную рекомендацию
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — ИИ-ассистент по спортивной психологии. "
                        "Отвечай только на вопросы, связанные с психологией спортсменов, "
                        "ментальной подготовкой, поддержкой родителей, отношениями с тренером и смежными темами. "
                        "Если тема не связана с этим — мягко уточни, что можешь говорить только об этом. "
                        "Начинай ответ с краткой рекомендации по запросу."
                    )
                },
                {"role": "user", "content": user_message}
            ]
        )

        base_reply = completion.choices[0].message.content

        # Подбор психологов
        matches = find_relevant_psychologists(user_message)
        if matches:
            base_reply += "\n\nПо вашему запросу могу порекомендовать следующих специалистов:"
            for match in matches:
                base_reply += f"\n\n👤 {match['name']} — {match['description']}\n🔗 {match['link']}"

        return jsonify({"response": base_reply})

    except Exception as e:
        print("Ошибка сервера:", str(e))
        return jsonify({"response": "Ошибка на сервере. Попробуйте позже."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
