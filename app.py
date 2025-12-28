from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import joblib
import os
import gdown

# ---------------- ENV ----------------
load_dotenv(".env.local")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key")

# ---------------- MODEL DOWNLOAD ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_pipeline.pkl")

GDRIVE_ID = "https://drive.google.com/file/d/1I5F3V0Tjk0Bl0GHHKJkideC0K5MSptNe/view?usp=sharing"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

if not os.path.exists(MODEL_PATH):
    print("⬇ Downloading ML model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

model = None
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model load failed:", e)

# ---------------- GEMINI ----------------
GEMINI_CHATBOX_API_KEY = os.getenv("GEMINI_CHATBOX_API_KEY")
genai.configure(api_key=GEMINI_CHATBOX_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- ROUTES ----------------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    prediction = None
    confidence = None
    news_text = ""
    word_count = 0

    session.setdefault("history", [])

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()

        if not news_text:
            prediction = "empty"

        elif model is None:
            prediction = "error"

        else:
            pred_value = int(model.predict([news_text])[0])
            prob = model.predict_proba([news_text])[0]

            prediction = "REAL" if pred_value == 0 else "FAKE"
            confidence = round(max(prob) * 100, 2)
            word_count = len(news_text.split())

            session["history"].insert(0, {
                "snippet": news_text[:35] + "...",
                "result": prediction,
                "conf": confidence
            })
            session["history"] = session["history"][:5]
            session.modified = True

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text,
        word_count=word_count,
        history=session.get("history", [])
    )

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_msg = request.json.get("message", "")

    try:
        response = gemini_model.generate_content(user_msg)
        return jsonify({"msg": response.text, "target": None})
    except Exception:
        return jsonify({"msg": "AI service unavailable.", "target": None})

@app.route("/clear_history")
def clear_history():
    session["history"] = []
    return redirect(url_for("dashboard"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
