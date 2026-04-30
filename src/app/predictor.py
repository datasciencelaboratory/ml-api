from app.model_loader import ModelLoader

class Predictor:

    def __init__(self):

        self.model = ModelLoader.load_model()

    def predict(self, text: str):

        prediction = self.model.predict([text])[0]

        probability = max(
            self.model.predict_proba([text])[0]
        )

        return {
            "label": prediction,
            "confidence": float(probability)
        }