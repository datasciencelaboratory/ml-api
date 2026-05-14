from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):

    message: str


class PredictResponse(BaseModel):

    label: str
    confidence: float

class NERRequest(BaseModel):
    message: str

class EntityOut(BaseModel):
    entity: str
    label: str

class NERResponse(BaseModel):
    parameters: List[EntityOut]

class IntentRequest(BaseModel):
    message: str

class IntentResponse(BaseModel):
    intent: str
    confidence: float


class Request(BaseModel):
    text: str

