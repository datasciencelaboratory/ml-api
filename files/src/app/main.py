from fastapi import FastAPI, HTTPException

from app.schemas import (
    PredictRequest,
    PredictResponse,
    NERRequest,
    NERResponse
)

from app.predictor import Predictor, extract_ner_parameters


from app.model_loader import load_ner_model
from contextlib import asynccontextmanager

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----------------------------------------------------
    # TUDO AQUI RODA ANTES DO SERVIDOR ACEITAR REQUISIÇÕES
    # ----------------------------------------------------
    print("Iniciando o servidor e carregando o modelo NER na memória...")
    try:
        # Carrega o modelo chamando a função que você já criou
        ml_models["ner_model"] = load_ner_model()
        print("Modelo NER carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo NER: {e}")
    
    # O 'yield' sinaliza que o servidor está pronto para rodar
    yield 
    
    # ----------------------------------------------------
    # TUDO AQUI RODA QUANDO O SERVIDOR FOR DESLIGADO
    # ----------------------------------------------------
    print("Limpando memória e desligando servidor...")
    ml_models.clear()

app = FastAPI(
    title="Intent Classifier API",
    version="1.0.0",
    lifespan=lifespan
)



predictor = Predictor()


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(request: PredictRequest):

    result = {}

    result['predict'] = predictor.predict(
        request.message
    )

    print("Resultado da predição:", result["predict"]["confidence"])

    if result["predict"]["label"] == 'command':
        print("Confiança baixa. Acionando extração NER...")
        
        # Recupera o modelo NER da memória RAM
        ner_model = ml_models.get("ner_model")
        
        if ner_model is None:
            raise HTTPException(status_code=503, detail="Modelo NER não carregado.")
        
        # Chama a função de extração que criamos
        ner_result = extract_ner_parameters(ner_model, request.message)
        result['ner'] = ner_result
        # Retorna o resultado do NER
        return result
    return result

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----------------------------------------------------
    # TUDO AQUI RODA ANTES DO SERVIDOR ACEITAR REQUISIÇÕES
    # ----------------------------------------------------
    print("Iniciando o servidor e carregando o modelo NER na memória...")
    try:
        # Carrega o modelo chamando a função que você já criou
        ml_models["ner_model"] = load_ner_model()
        print("Modelo NER carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo NER: {e}")
    
    # O 'yield' sinaliza que o servidor está pronto para rodar
    yield 
    
    # ----------------------------------------------------
    # TUDO AQUI RODA QUANDO O SERVIDOR FOR DESLIGADO
    # ----------------------------------------------------
    print("Limpando memória e desligando servidor...")
    ml_models.clear()



@app.post("/ner", response_model=NERResponse)
async def ner_endpoint(request: NERRequest):
    # Em vez de carregar do disco, pegamos o modelo instantaneamente da memória
    ner_model = ml_models.get("ner_model")
    
    if ner_model is None:
        raise HTTPException(status_code=503, detail="O modelo NER não foi carregado corretamente.")
    
    try:
        # Executa a predição de forma muito mais rápida
        result = extract_ner_parameters(ner_model, request.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na extração: {str(e)}")