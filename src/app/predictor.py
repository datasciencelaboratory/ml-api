from app.model_loader import ModelLoader
import torch
import torch.nn.functional as F

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

def predict_intent(model_bundle, message: str):
    tokenizer, model, device = model_bundle

    #tokenizacao
    inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)
        confidence, class_idx = torch.max(probabilities, dim=-1)

        #mapeia ids para labels

        intent_label = model.config.id2label[class_idx.item()]

        return {
            "tool": intent_label,
            "confidence" : float(confidence.item())
        }
