import joblib
from pathlib import Path
import os
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from llama_cpp import Llama

GGUF_MODEL_PATH = "model/qwen_q8.gguf" # Substitua pelo nome real do seu arquivo
MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "intent_model_v2.pkl"
NER_MODEL_PATH = "model/ner_parameters"

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
    model_path = "model/modelo_intencaov2"

    #carrega o tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    #coloca o modelo em evaluate mode e mova para cpu
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return tokenizer, model, device




def load_gguf_model():
    """Carrega o modelo GGUF usando llama-cpp-python."""
    if not os.path.exists(GGUF_MODEL_PATH):
        raise FileNotFoundError(f"Modelo GGUF não encontrado em: {GGUF_MODEL_PATH}")
    
    print("Carregando modelo GGUF...")
    # O parâmetro n_ctx define o tamanho do contexto. Ajuste conforme a necessidade do seu modelo.
    model = Llama(model_path=GGUF_MODEL_PATH, n_ctx=1024, verbose=False)
    return model