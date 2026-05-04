from pydantic import BaseModel


class PredictRequest(BaseModel):

    message: str


class PredictResponse(BaseModel):

    label: str
    confidence: float