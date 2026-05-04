import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "model" / "intent_model_v2.pkl"

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