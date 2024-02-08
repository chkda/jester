from typing import Dict, Any


import ray 
from ray import serve
from starlette.requests import Request
from jester.predictor import LLMPredictor

@serve.deployment(num_replicas=1)
class ModelServer:

    def __init__(self,config: Dict[str, Any]):
        self.dummy_response = "Hello world"
        self.predictor = LLMPredictor(config)

    def compute(self) -> str:
        output = self.predictor.predict()
        return output

    def reconfigure(self, config: Dict[str, Any]):
        self.predictor = LLMPredictor(config)

    async def __call__(self, request: Request) -> str: 
        response = self.compute()
        return "Hello world " + response

app = ModelServer.bind(config={"engine":"hf"})