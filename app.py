from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import joblib
import os

# -------------------- ENV SETUP --------------------
load_dotenv(".env.local")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key")

# -------------------- MODEL LOADING --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_pipeline.pkl")

model = None
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)

# -------------------- GEMINI CONFIG --------------------
GEMINI_CHATBOX_API_KEY = os.getenv("GEMINI_CHATBOX_API_KEY")
genai.configure(api_key=GEMINI_CHATBOX_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------- ROUTES --------------------
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
            probability = model.predict_proba([news_text])[0]

            prediction = "REAL" if pred_value == 0 else "FAKE"
            confidence = round(max(probability) * 100, 2)
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

    system_context = (
    "You are an expert Fake News Detector AI for a student NLP project called NoCap. "
    "Your core task is to analyze, explain, and answer questions related ONLY to fake news detection, "
    "news verification, misinformation, and how the system works. "
    "The system uses Natural Language Processing and Logistic Regression to classify news as REAL or FAKE. "
    
    "You are allowed to answer:\n"
    "- How fake news is detected\n"
    "- How Logistic Regression and NLP are used\n"
    "- Questions about news credibility\n"
    "- Questions related to the system features and directories (Workspace, Intelligence)\n"
    
    "Rules:\n"
    "- If the question is NOT related to news, fake news detection, or the system directory, "
    "politely refuse and say you can only answer fake-news-related questions.\n"
    "- Keep answers short, clear, and technical.\n"
    "- If the user wants to analyze or scan news, mention 'Workspace'.\n"
    "- If the user wants model, dataset, or system explanation, mention 'Intelligence'."
    )

    try:
        response = gemini_model.generate_content(
            f"{system_context}\nUser: {user_msg}"
        )

        reply = response.text.lower()
        target = None
        if "workspace" in reply:
            target = "workspace-section"
        elif "intelligence" in reply:
            target = "info-section"

        return jsonify({"msg": response.text, "target": target})

    except Exception:
        return jsonify({"msg": "AI service unavailable.", "target": None})

@app.route("/clear_history")
def clear_history():
    session["history"] = []
    session.modified = True
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
