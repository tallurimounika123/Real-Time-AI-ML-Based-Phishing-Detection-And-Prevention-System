from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

model = joblib.load("model.pkl")

# Store last 10 scan records
history = []

# Feature extraction
def extract(text):
    return [[
        len(text),
        text.count('.'),
        int('@' in text),
        int('-' in text),
        int('https' in text),
        int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', text))),
        int('login' in text),
        int('verify' in text),
        int('secure' in text)
    ]]

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""
    show_history = False

    global history

    # If History button clicked
    if request.method == "POST" and "history" in request.form:
        show_history = True
        return render_template("index.html",
                               history=history,
                               show_history=show_history)

    if request.method == "POST":
        data = ""

        # URL input
        if request.form.get("url"):
            data = request.form.get("url")

        # File upload
        if 'file' in request.files:
            f = request.files['file']
            if f.filename != "":
                data = f.read().decode(errors="ignore")

        if data:
            features = extract(data)

            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0]
            confidence = round(max(prob) * 100, 2)

            if pred == 1:
                result = "Phishing"
                display_result = "🚨 Phishing Detected"
            else:
                result = "Legitimate"
                display_result = "✅ Legitimate"

            # Store only URL scans in history
            if request.form.get("url"):
                record = {
                    "url": data,
                    "result": result,
                    "confidence": confidence
                }

                history.insert(0, record)

                # Keep only last 10 records
                if len(history) > 10:
                    history = history[:10]

            return render_template("index.html",
                                   result=display_result,
                                   confidence=confidence,
                                   history=history,
                                   show_history=False)

    return render_template("index.html",
                           history=history,
                           show_history=show_history)

if __name__ == "__main__":
    app.run(debug=True)