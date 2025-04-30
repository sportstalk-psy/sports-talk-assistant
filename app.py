from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("psychologists_base.json", "r", encoding="utf-8") as f:
    psychologists = json.load(f)

def get_embedding(text):
    return np.array(
        client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    )

def find_relevant_psychologists(query, top_n=2, threshold=0.3):
    query_embedding = get_embedding(query).reshape(1, -1)
    results = []
    for person in psychologists:
        try:
            desc_embedding = get_embedding(person["description"]).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, desc_embedding)[0][0]
            if similarity >= threshold:
                results.append((person, similarity))
        except Exception as e:
            print(f"⚠️ Ошибка эмбеддинга у {person['name']}: {e}")
    relevant = sorted(results, key=lambda x: -x[1])
    return [r[0] for r in relevant[:top_n]]

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"response": "Пожалуйста, напишите сообщение."})

        messages = [
            {"role": "system", "content": (
                "Ты — ИИ-ассистент платформы спортивной психологии Sports Talk, тебя зовут Марк. "
                "Отвечай дружелюбно, ободряющим тоном, с опорой на мотивацию и будущее. "
                "Если тебя спрашивают предложить психолога — выбирай только из приложенного JSON-файла, не пиши имена и фамилии психологов в общей рекомендации и те, которых нет в файле JSON."
            )},
            {"role": "user", "content": user_message}
        ]

        completion = client.chat.completions.create(model="gpt-4-turbo", messages=messages)
        base_reply = completion.choices[0].message.content.strip()

        matches = find_relevant_psychologists(user_message)
        if matches:
            base_reply += "\n\nПо вашему запросу могу порекомендовать специалистов:"
            for match in matches:
                base_reply += (
                    f"<br><br><strong>👤 {match['name']}</strong><br>"
                    f"{match['description']}<br>"
                    f"<a href='{match['link']}' target='_blank'>Посмотреть профиль</a>"
                )

        return jsonify({"response": base_reply})

    except Exception as e:
        print("🚨 Ошибка сервера:", str(e))
        return jsonify({"response": "Произошла ошибка на сервере. Попробуйте позже."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)