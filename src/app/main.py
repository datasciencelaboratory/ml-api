from fastapi import FastAPI, HTTPException
import json
from app.schemas import (
    PredictRequest,
    NERRequest,
    NERResponse,
    IntentRequest, 
    IntentResponse,
    ClassifyRequest,
    ClassifyResponse,
    Request
)

from app.predictor import extract_ner_parameters, predict_intent


from app.model_loader import load_ner_model, load_intent_model, load_gguf_model
from contextlib import asynccontextmanager

from app.utils import extrair_entidades_regex

import os
from functools import lru_cache

import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando o servidor e carregando os modelos na memória...")
    
    # 1. Tentativa de carregar o Predictor clássico
    try:
        from app.predictor import Predictor
        ml_models["predictor"] = Predictor()
        print("Predictor clássico carregado com sucesso!")
    except Exception as e:
        print(f"Aviso: Não foi possível carregar o Predictor: {e}")
        ml_models["predictor"] = None

    # 2. Tentativa de carregar o modelo NER
    try:
        ml_models["ner_model"] = load_ner_model()
        print("Modelo NER carregado com sucesso!")
    except Exception as e:
        print(f"Aviso: Não foi possível carregar o modelo NER: {e}")
        ml_models["ner_model"] = None
    
    # 3. Tentativa de carregar o modelo Intent (Transformers)
    try:
        ml_models["intent_bundle"] = load_intent_model()
        print("Modelo Intent carregado com sucesso!")
    except Exception as e:
        print(f"Aviso: Não foi possível carregar o modelo Intent: {e}")
        ml_models["intent_bundle"] = None

    # 4. Tentativa de carregar o modelo GGUF (Llama-cpp)
    try:
        from app.model_loader import load_gguf_model
        ml_models["gguf_model"] = load_gguf_model()
        print("Modelo GGUF carregado com sucesso!")
    except Exception as e:
        print(f"Aviso: Não foi possível carregar o modelo GGUF: {e}")
        ml_models["gguf_model"] = None
    
    yield 
    
    print("Limpando memória e desligando servidor...")
    ml_models.clear()

app = FastAPI(
    title="Intent Classifier API",
    version="1.0.0",
    lifespan=lifespan
)



@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(request: PredictRequest):

    result = {}

    result['predict'] = ml_models['predictor'].predict(
        request.message
    )

    print("Resultado da predição:", result["predict"]["confidence"])

    if result["predict"]["label"] == 'command':
        print("Confiança baixa. Acionando extração NER...")
        
        # Recupera o modelo NER da memória RAM
        ner_model = ml_models.get("ner_model")
        
        if ner_model is None:
            raise HTTPException(status_code=503, detail="Modelo NER não carregado.")
        
        if ml_models['predictor'] is None:
            raise HTTPException(status_code=503, detail="Modelo Intent não carregado.")
        
        # Chama a função de extração que criamos
        ner_result = extract_ner_parameters(ner_model, request.message)
        prediction = predict_intent(ml_models['predictor'], request.message)
        result['intent'] = prediction
        result['ner'] = ner_result
        
        result['ner'] = extrair_entidades_regex(request.message, result['ner'])

        # Retorna o resultado do NER
        return result
    return result



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

@app.post('/intent_predict', response_model=IntentResponse)
async def get_intent(request: IntentRequest):
    #passa o bundle completo para o predictor
    prediction = predict_intent(ml_models['predictor'], request.message)
    return prediction



@lru_cache(maxsize=1)
def _bm25_model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name="Qdrant/bm25")


@app.post("/embed")
def embed(req: Request):
    result = genai.embed_content(
        model="models/gemini-embedding-2-preview",
        content=req.text,
        task_type="retrieval_document",
    )

    return {"embedding": result["embedding"]}


@app.post("/sparse")
def sparse(req: Request):
    """BM25 esparso (fastembed Qdrant/bm25) no formato Qdrant: indices + values."""
    model = _bm25_model()
    sparse_emb = next(model.embed([req.text]))
    indices = sparse_emb.indices
    values = sparse_emb.values
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    else:
        indices = list(indices)
    if hasattr(values, "tolist"):
        values = values.tolist()
    else:
        values = list(values)
    return {"indices": indices, "values": values}



@app.post("/classify")
async def classify_endpoint(request: ClassifyRequest):
    gguf_model = ml_models.get("gguf_model")
    if gguf_model is None:
        raise HTTPException(status_code=503, detail="O modelo GGUF para /classify não está disponível.")
    
    # 1. Definimos o comportamento esperado (igual ao seu dataset de treino)
    system_prompt = (
        "Você é uma inteligência artificial especializada em análise de linguagem natural "
        "para sistemas de controle de acesso de condomínios e empresas.\n"
        "Sua tarefa é classificar a intenção (Intent) do usuário e extrair entidades (NER).\n"
        "Sempre responda APENAS com um JSON válido contendo as chaves: intent, parameters, entities e response."
    )
    
    # 2. Montamos o prompt EXATAMENTE no formato ChatML do seu fine-tuning
    prompt_formatado = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{request.message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    try:
        # 3. Executamos o modelo
        output = gguf_model(
            prompt_formatado,
            max_tokens=300,        # Aumentado para suportar o tamanho do JSON gerado
            stop=["<|im_end|>"],   # CRÍTICO: Faz o modelo parar ao terminar de gerar o JSON
            echo=False,            # Não repete o prompt na resposta
            temperature=0.1        # Temperatura baixa (0.1) garante saídas mais precisas e JSONs mais estáveis
        )
        
        # Extrai o texto gerado
        resultado_texto = output["choices"][0]["text"].strip()
        
        # 4. (Opcional, mas recomendado) Valida se a resposta é realmente um JSON
        try:
            resultado_json = json.loads(resultado_texto)
            return resultado_json # Retorna o JSON direto para quem chamou a API
        except json.JSONDecodeError:
            # Fallback caso o modelo tenha alucinado e gerado texto fora do JSON
            return {"erro": "O modelo não retornou um JSON válido", "texto_bruto": resultado_texto}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência GGUF: {str(e)}")