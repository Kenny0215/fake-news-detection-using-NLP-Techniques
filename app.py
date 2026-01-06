from flask import Flask, request, render_template, session, redirect, url_for, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import joblib
import os

load_dotenv(".env.local")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- MODEL LOADING ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fake_news_pipeline.pkl")
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Load model once when app starts
load_model()

# --- GEMINI CONFIGURATION ---
GEMINI_CHATBOX_API_KEY = os.getenv("GEMINI_CHATBOX_API_KEY")
genai.configure(api_key=GEMINI_CHATBOX_API_KEY)

gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# --- HELPER FUNCTION FOR PREDICTION ---
def get_prediction(text):
    if model is None:
        return None, None
    try:
        pred_value = int(model.predict([text])[0])
        probability = model.predict_proba([text])[0]
        confidence = round(max(probability) * 100, 2)
        return pred_value, confidence
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None, None

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
        if news_text:
            # Using the helper function
            prediction, confidence = get_prediction(news_text)
            word_count = len(news_text.split())
            
            if word_count < 10:
                prediction = "too_short" # Set a special flag
            else:
                # Only analyze if 10 words or more
                prediction, confidence = get_prediction(news_text)

            if prediction is not None and prediction != "too_short":
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

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Handles smart chat and real-time fake news detection within the chat."""
    user_msg = request.json.get("message", "")
    
    # 1. RUN THE LOCAL MODEL FIRST
    local_pred, local_conf = get_prediction(user_msg)
    
    # Create a status string to tell Gemini what our model found
    model_insight = ""
    if local_pred is not None and len(user_msg.split()) > 5:
        result_str = "FAKE" if local_pred == 1 else "REAL"
        model_insight = f" (NOTE: Our local NLP model analyzed this text and is {local_conf}% sure it is {result_str}.)"

    # 2. UPDATE SYSTEM CONTEXT
    system_context = (
        "You are an expert Fake News Detector AI named NoCap. "
        "Your task is to analyze news credibility and explain system features. "
        "If a user provides a news article or claim in the chat, use the provided 'Model Insight' to give them a verdict. "
        "Always explain WHY a piece of news might be flagged (e.g., look for sensationalism, lack of sources, or clickbait patterns)."
        "\n\nSystem Details: Uses NLP & Logistic Regression. Directories: Workspace (for main scanning), Intelligence (for tech details)."
        "\n\nRules: "
        "- Be professional but direct."
        "- If the text is unrelated to news or the project, politely stay on topic."
        "- If 'Model Insight' is provided, use it to guide your answer."
    )

    try:
        # 3. PASS EVERYTHING TO GEMINI
        full_prompt = f"{system_context}\nModel Insight: {model_insight}\nUser says: {user_msg}"
        
        response = gemini_model.generate_content(full_prompt)
        bot_reply = response.text
        
        # 4. AUTO-REDIRECT LOGIC
        target = None
        reply_lower = bot_reply.lower()
        if any(x in reply_lower for x in ["workspace", "detect", "paste"]):
            target = "workspace-section"
        elif any(x in reply_lower for x in ["intelligence", "about", "team", "logic"]):
            target = "info-section"

        return jsonify({"msg": bot_reply, "target": target})
    except Exception as e:
        print(f"Chatbot Error: {e}")
        return jsonify({"msg": "I encountered an error analyzing that. Please try the main Workspace detector!", "target": None})

@app.route("/clear_history")
def clear_history():
    session['history'] = []
    session.modified = True
    return redirect(url_for('index'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)