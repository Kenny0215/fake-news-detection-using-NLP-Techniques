from flask import Flask, request, render_template, session, redirect, url_for
import joblib
import os

app = Flask(__name__)
app.secret_key = os.urandom(24) 

try:
    model = joblib.load("fake_news_pipeline.pkl")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    news_text = ""
    word_count = 0
    
    # Initialize history session
    if 'history' not in session:
        session['history'] = []

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        if news_text:
            # Model Processing
            pred_value = int(model.predict([news_text])[0])
            probability = model.predict_proba([news_text])[0]
            
            prediction = pred_value
            confidence = round(max(probability) * 100, 2)
            word_count = len(news_text.split())

            # Store in History (Last 5 items)
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

@app.route("/clear_history")
def clear_history():
    session['history'] = []
    session.modified = True
    # Redirecting back to "/" prevents the 405 error on the next scan
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)