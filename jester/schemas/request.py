from pydantic import BaseModel

class Request(BaseModel):
    prompt: str
    temperature: float = 0.7