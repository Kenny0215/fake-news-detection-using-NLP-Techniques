import os
from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import joblib

# Load environment variables
load_dotenv() 

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

# --- MODEL LOADING ---
MODEL_PATH = "fake_news_pipeline.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("SUCCESS: Model loaded successfully.")
    else:
        print(f"ERROR: {MODEL_PATH} not found in root directory!")
except Exception as e:
    print(f"ERROR loading model: {e}")

# --- GEMINI CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_CHATBOX_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("WARNING: GEMINI_CHATBOX_API_KEY is missing!")

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/dashboard", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    news_text = ""
    word_count = 0
    
    if 'history' not in session:
        session['history'] = []

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        
        # SAFETY CHECK: If model didn't load, don't crash the server
        if model is None:
            return "Server Error: ML Model file is missing or corrupted. Check Render logs.", 500

        if news_text:
            try:
                # Run prediction
                pred_value = int(model.predict([news_text])[0])
                probability = model.predict_proba([news_text])[0]
                
                prediction = pred_value
                confidence = round(max(probability) * 100, 2)
                word_count = len(news_text.split())

                new_entry = {
                    "snippet": news_text[:35] + "...", 
                    "result": "REAL" if prediction == 0 else "FAKE",
                    "conf": confidence
                }
                history_list = session.get('history', [])
                history_list.insert(0, new_entry)
                session['history'] = history_list[:5]
                session.modified = True 
            except Exception as e:
                print(f"Prediction Error: {e}")
                prediction = "error"
        else:
            prediction = "empty"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text,
        word_count=word_count,
        history=session.get('history', [])
    )

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_msg = request.json.get("message", "")
    system_context = "You are an expert Fake News Detector AI..."

    try:
        response = gemini_model.generate_content(f"{system_context} User: {user_msg}")
        bot_reply = response.text
        return jsonify({"msg": bot_reply, "target": None})
    except Exception as e:
        return jsonify({"msg": "AI Chat currently unavailable.", "target": None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)