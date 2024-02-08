from typing import Dict, Any

from jester.llm.vllm import VLLMEngine
from jester.llm.ggml import GGMLEngine
from jester.llm.hugging_face import HFEngine


class LLMPredictor:

    def __init__(self, config: Dict[str, Any]):
        self.engine = None
        if config['engine'] == "hf":
            self.engine = HFEngine()
        elif config['engine'] == "vllm":
            self.engine = VLLMEngine()
        elif config['engine'] == "ggml":
            self.engine = GGMLEngine()
        else:
            raise RuntimeError("Unknown engine type")

    def predict(self):
        return self.engine.run("dummy input")