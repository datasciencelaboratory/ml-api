import joblib
from pathlib import Path
import os
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "intent_model_v2.pkl"
NER_MODEL_PATH = "files/model/ner_parameters"

class ModelLoader:

    _model = None

    @classmethod
    def load_model(cls):

        if cls._model is None:

            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Modelo não encontrado em {MODEL_PATH}"
                )

            cls._model = joblib.load(MODEL_PATH)

        return cls._model

def load_ner_model():
    """Carrega o modelo NER do diretório local."""
    if not os.path.exists(NER_MODEL_PATH):
        raise FileNotFoundError(f"Pasta do modelo não encontrada em: {NER_MODEL_PATH}")
    
    print("Carregando modelo NER...")
    # Assumindo que seja um modelo spaCy treinado
    model = spacy.load(NER_MODEL_PATH)
    return model


def load_intent_model():
    model_path = "files/model/modelo_intencao"

    #carrega o tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    #coloca o modelo em evaluate mode e mova para cpu
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return tokenizer, model, device