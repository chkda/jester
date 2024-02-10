import asyncio
from queue import Empty
from starlette.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

class HFEngine:

    def __init__(self, model_path: str):
        self.loop = asyncio.get_running_loop()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

    def run(self, input):
        return "From HF"

    def stream_response(self, prompt: str) -> StreamingResponse:
        streamer = TextIteratorStreamer(self.tokenizer, timeout=0, 
                    skip_special_tokens=True, skip_prompt=True)
        self.loop.run_in_executor(None, self.generate_text, prompt, streamer)
        response = StreamingResponse(self.consume_stream(streamer), media_type="text/plain")
        return response

    def generate_text(self, prompt: str, streamer: TextIteratorStreamer):
        input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
        self.model.generate(input_ids, streamer=streamer, max_length=10000, pad_token_id=self.tokenizer.eos_token_id)

    async def consume_stream(self, streamer: TextIteratorStreamer):
        while True:
            try:
                for token in streamer:
                    print("Toks:", token)
                    yield token
                break
            except Empty:
                await asyncio.sleep(0.001)
