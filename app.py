from flask import Flask, request, render_template
import joblib
import re
import string
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

models = {
    'random_forest': joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl')),
    'logistic': joblib.load(os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')),
    'xgboost': joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl')),
    'svc': joblib.load(os.path.join(MODELS_DIR, 'svc_model.pkl')),
    'stacking': joblib.load(os.path.join(MODELS_DIR, 'stacking_classifier_model.pkl')),
    'naive_bayes': joblib.load(os.path.join(MODELS_DIR, 'naivebayes_model.pkl'))
}

vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))

# Preprocessing function
def preprocess_email(email: str) -> str:
    email = email.lower()
    email = re.sub(r"http\S+|www\S+|https\S+", '', email, flags=re.MULTILINE)
    email = re.sub(r'\d+', '', email)
    email = email.translate(str.maketrans('', '', string.punctuation))
    email = re.sub(r'\s+', ' ', email).strip()
    return email

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    probability = None
    chosen_model = 'logistic'  # default

    if request.method == 'POST':
        email_text = request.form.get('email', '').strip()
        chosen_model = request.form.get('model', chosen_model)

        if email_text and chosen_model in models:
            cleaned = preprocess_email(email_text)
            features = vectorizer.transform([cleaned])
            model = models[chosen_model]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0][1]
            else:
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
    app.debug = True
    app.run()