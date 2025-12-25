from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import joblib
import os

load_dotenv(".env.local")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- GEMINI CONFIGURATION ---
GEMINI_CHATBOX_API_KEY = os.getenv("GEMINI_CHATBOX_API_KEY")
genai.configure(api_key=GEMINI_CHATBOX_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

try:
    model = joblib.load("fake_news_pipeline.pkl")
except Exception as e:
    print(f"Error loading model: {e}")

# --- ROUTE 1: LANDING PAGE ---
@app.route("/")
def landing():
    """This renders the professional intro page first."""
    return render_template("landing.html")

# --- ROUTE 2: DETECTION DASHBOARD ---
@app.route("/dashboard", methods=["GET", "POST"])
def index():
    """This handles the main fake news detection logic."""
    prediction = None
    confidence = None
    news_text = ""
    word_count = 0
    
    if 'history' not in session:
        session['history'] = []

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        if news_text:
            # Predict using the loaded model
            pred_value = int(model.predict([news_text])[0])
            probability = model.predict_proba([news_text])[0]
            
            prediction = pred_value
            confidence = round(max(probability) * 100, 2)
            word_count = len(news_text.split())

            # Update History (Keep last 5)
            new_entry = {
                "snippet": news_text[:35] + "...", 
                "result": "REAL" if prediction == 0 else "FAKE",
                "conf": confidence
            }
            history_list = session.get('history', [])
            history_list.insert(0, new_entry)
            session['history'] = history_list[:5]
            session.modified = True 
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

# --- ROUTE 3: GEMINI CHATBOT ---
@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Handles smart chat and automatic UI redirection."""
    user_msg = request.json.get("message", "")
    
    # Context tells Gemini who it is and how to behave
    system_context = (
        "You are the NoCap Assistant for a student NLP project. "
        "The project uses Logistic Regression to detect fake news. "
        "Keep answers short. If the user wants to scan news, mention 'Workspace'. "
        "If they want project details, mention 'Intelligence'."
    )

    try:
        response = gemini_model.generate_content(f"{system_context} User: {user_msg}")
        bot_reply = response.text
        
        # Determine if the bot should trigger a scroll on the UI
        target = None
        reply_lower = bot_reply.lower()
        if any(x in reply_lower for x in ["workspace", "detect", "paste"]):
            target = "workspace-section"
        elif any(x in reply_lower for x in ["intelligence", "about", "team", "logic"]):
            target = "info-section"

        return jsonify({"msg": bot_reply, "target": target})
    except Exception as e:
        return jsonify({"msg": "Connection to AI failed. Try again!", "target": None})

# --- ROUTE 4: CLEAR HISTORY ---
@app.route("/clear_history")
def clear_history():
    session['history'] = []
    session.modified = True
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)