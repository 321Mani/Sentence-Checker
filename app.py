from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def home():
    return "API Running"

@app.route('/check', methods=['POST'])
def check():
    data = request.json
    new_issue = data['issue']
    existing = data['existing']

    new_vec = model.encode(new_issue)

    scores = []
    for e in existing:
        score = util.cos_sim(new_vec, model.encode(e))
        scores.append(float(score))

    return jsonify({"max_score": max(scores)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)