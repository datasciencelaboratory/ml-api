from fastapi import FastAPI

from app.schemas import (
    PredictRequest,
    PredictResponse
)

from app.predictor import Predictor


app = FastAPI(
    title="Intent Classifier API",
    version="1.0.0"
)

predictor = Predictor()


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    result = predictor.predict(
        request.message
    )

    return result