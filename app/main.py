import pickle
import re
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer 

app = FastAPI()
vectorizer = TfidfVectorizer()


with open("./final_model.pkl", 'rb') as f:
    model = pickle.load(f)

def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def vectorize_query(text):
    query_vector = vectorizer.transform([text])
    return query_vector

@app.get("/")
def get_api_info():
    return {"api_version": "1.0.0"}

@app.post("/predict")
def predict_text(text):
    text = preprocess_text(text)
    query = vectorize_query(text)
    prediction = model.predict(query)
    prediction = str(prediction[0])
    return {"predicted_label": prediction}