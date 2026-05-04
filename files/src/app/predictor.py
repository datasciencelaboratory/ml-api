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
    
def extract_ner_parameters(model, message: str):
    """
    Recebe o texto e o modelo carregado, retornando as entidades.
    """
    doc = model(message)
    
    extracted_parameters = []
    for ent in doc.ents:
        extracted_parameters.append({
            "entity": ent.text,
            "label": ent.label_
        })
        
    return {"parameters": extracted_parameters}