import numpy as np
import torch
from transformers import pipeline
from fastapi import FastAPI, HTTPException, Body

app = FastAPI(debug=True)  # Enable debug mode

# Load the pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@app.post("/predict/")
async def predict(text: str = Body(...)):
    try:
        # Call the sentiment analysis pipeline with the provided text
        prediction = sentiment_pipeline(text)
        
        # Extract sentiment and confidence from the prediction
        sentiment = prediction[0]['label']
        confidence = prediction[0]['score']
        
        return {"text": text, "sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
