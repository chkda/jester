from typing import Dict, Any

from jester.llm.vllm import VLLMEngine
from jester.llm.ggml import GGMLEngine
from jester.llm.hugging_face import HFEngine


class LLMPredictor:

    def __init__(self, config: Dict[str, Any]):
        self.engine = None
        self.config = config
        if self.config['engine'] == "hf":
            self.engine = HFEngine(model_path=config['model_path'])
        elif self.config['engine'] == "vllm":
            self.engine = VLLMEngine()
        elif self.config['engine'] == "ggml":
            self.engine = GGMLEngine()
        else:
            raise RuntimeError("Unknown engine type")

    def predict(self, prompt: str):
        if self.config['stream']:
            return self.engine.stream_response(prompt)
        return self.engine.run("dummy input")