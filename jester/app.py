from typing import Dict, Any
from starlette.responses import StreamingResponse
import ray 
from ray import serve
from jester.schemas.request import Request
from jester.predictor import LLMPredictor


@serve.deployment
class ModelServer:

    def __init__(self,config: Dict[str, Any]):
        self.dummy_response = "Hello world"
        self.predictor = LLMPredictor(config)

    def compute(self, request: Dict[str, Any]) -> StreamingResponse:
        prompt = request['prompt']
        # output = self.predictor.predict()
        return self.predictor.predict(prompt)

    def reconfigure(self, config: Dict[str, Any]):
        self.predictor = LLMPredictor(config)

    async def __call__(self, request: Request) -> StreamingResponse: 
        req_dict = await request.json()
        response = self.compute(req_dict)
        return response

app = ModelServer.bind(config={"engine":"hf", "model_path":"microsoft/DialoGPT-small"})