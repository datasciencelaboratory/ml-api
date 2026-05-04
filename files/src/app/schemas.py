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