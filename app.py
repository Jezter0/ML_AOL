from flask import Flask, request, render_template
import joblib
import re
import string
import os

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load models and vectorizer
models = {
    'random_forest': joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl')),
    'logistic': joblib.load(os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')),
    'xgboost': joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl')),
    'svc': joblib.load(os.path.join(MODELS_DIR, 'svc_model.pkl')),
}

vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))

# Preprocessing function
def preprocess_email(email: str) -> str:
    email = email.lower()
    email = re.sub(f"[{string.punctuation}]", " ", email)
    email = re.sub(r"\s+", " ", email).strip()
    return email

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    probability = None
    chosen_model = 'random_forest'  # default

    if request.method == 'POST':
        email_text = request.form.get('email', '').strip()
        chosen_model = request.form.get('model', chosen_model)

        if email_text and chosen_model in models:
            cleaned = preprocess_email(email_text)
            features = vectorizer.transform([cleaned])
            model = models[chosen_model]

            # Handle SVC separately if needed
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0][1]
            else:
                # For models without predict_proba like some SVCs
                prediction = model.predict(features)[0]
                proba = 1.0 if prediction == 1 else 0.0

            is_spam = proba > 0.5
            result = "Spam" if is_spam else "Not Spam"
            probability = round(proba * 100, 2)

    return render_template(
        'index.html',
        models=models.keys(),
        result=result,
        probability=probability,
        chosen_model=chosen_model
    )

if __name__ == '__main__':
    app.run(debug=True)