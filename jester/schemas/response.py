from pydantic import BaseModel

class LLMResponse(BaseModel):
    text: str
    prompt: str
    temperature: float