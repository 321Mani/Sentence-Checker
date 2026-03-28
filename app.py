from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import os

app = Flask(__name__)

# ✅ Load model ONCE (important)
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

@app.route('/')
def home():
    return "API Running 🚀"

@app.route('/check', methods=['POST'])
def check():
    data = request.json
    new_issue = data['issue']
    existing = data['existing']

    # ✅ Encode once
    new_vec = model.encode(new_issue)

    # ✅ Encode all existing at once (FAST)
    existing_vecs = model.encode(existing)

    # ✅ Compute similarity in one shot
    scores = util.cos_sim(new_vec, existing_vecs)[0]

    max_score = float(max(scores))

    return jsonify({
        "max_score": max_score
    })

# ✅ IMPORTANT: Railway dynamic port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
