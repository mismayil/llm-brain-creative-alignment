import argparse
import math
import os
from tqdm import tqdm
import pathlib
from abc import ABC, abstractmethod
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import asyncio, dataclasses
from dotenv import load_dotenv
import logging, sys
from google import genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, DeadlineExceeded
from google.genai import types as genai_types

logging.basicConfig(stream=sys.stderr, level=logging.WARN)
logger = logging.getLogger(__name__)

from cadabra.utils import read_json, write_json, generate_unique_id, generate_datetime_id, batched, none_or_int, none_or_str

def is_thinking_model(model_name):
    return model_name.startswith("gpt-5") or model_name.startswith("gemini-2.5") or model_name.startswith("gemini-3")

@dataclasses.dataclass
class ModelResponse:
    sample_id: str = None
    text: str = None
    usage: dict = None
    exception: Exception = None
    thoughts: str = None

class InferenceService(ABC):
    @abstractmethod
    async def generate(self, batch, **kwargs):
        pass

    @abstractmethod
    def prepare_model_args(self, model_args, **kwargs):
        pass

class GoogleService(InferenceService):
    def __init__(self, model_name, api_key, model_args=None):
        self.model_name = model_name
        self.api_key = api_key
        self.config = self.prepare_model_args(model_args)
        self.client = self.init_client()

    def init_client(self):
        return genai.Client(api_key=self.api_key if self.api_key is not None else os.getenv("GOOGLE_API_KEY"))

    def prepare_model_args(self, model_args):
        cfg = {}

        if model_args is not None:
            temp = model_args.get("temperature")
            if temp is not None and not is_thinking_model(self.model_name):
                cfg["temperature"] = temp

            max_tok = model_args.get("max_tokens")
            if max_tok is not None:
                # GenAI SDK uses max_output_tokens
                cfg["max_output_tokens"] = max_tok

            top_p = model_args.get("top_p")
            if top_p is not None:
                cfg["top_p"] = top_p

            top_k = model_args.get("top_k")
            if top_k is not None:
                cfg["top_k"] = top_k

            thinking_cfg = {"include_thoughts": model_args.get("enable_thinking_summary", False)}
            thinking_effort = model_args.get("enable_thinking_effort")
            
            if thinking_effort:
                if thinking_effort in ["minimal", "low", "medium", "high"]:
                    if self.model_name.startswith("gemini-2"):
                        thinking_cfg["thinking_budget"] = -1
                    else:
                        thinking_cfg["thinking_level"] = thinking_effort
                else:
                    raise ValueError(f"Invalid thinking effort level: {model_args.get('enable_thinking_effort')}")
            
            cfg["thinking_config"] = genai_types.ThinkingConfig(**thinking_cfg)

        return cfg

    def _extract_text_and_thoughts_from_response(self, response):
        text = ""
        thoughts = ""

        try:
            text_parts = []
            thought_parts = []
            for cand in getattr(response, "candidates", []) or []:
                content = getattr(cand, "content", None)
                if not content:
                    continue
                for part in (getattr(content, "parts", []) or []):
                    thought = getattr(part, "thought", None)
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        if thought:
                            thought_parts.append(txt.strip())
                        else:
                            text_parts.append(txt.strip())
            if text_parts:
                text = "\n".join(text_parts).strip()
            if thought_parts:
                thoughts = "\n".join(thought_parts).strip()
        except Exception:
            pass
        return text, thoughts

    def _extract_exception_from_response(self, response):
        exception = None
        finish_reason = None
        safety_ratings = None

        try:
            cands = getattr(response, "candidates", None) or []
            if cands:
                fr = getattr(cands[0], "finish_reason", None)
                if fr is not None:
                    finish_reason = str(fr)

                sr = getattr(cands[0], "safety_ratings", None)
                if sr is not None:
                    safety_ratings = ",".join([str(x) for x in sr])
        except Exception:
            pass
        
        if finish_reason and finish_reason != "FinishReason.STOP":
            exception = f"Finish reason: {finish_reason}, safety ratings: {safety_ratings}"
        
        return exception

    def _extract_usage_from_response(self, response):
            usage = {}

            try:
                usage_metadata = getattr(response, "usage_metadata", None)
                if usage_metadata is not None:
                    usage["input_tokens"] = getattr(usage_metadata, "prompt_token_count", None)
                    usage["output_tokens"] = getattr(usage_metadata, "candidates_token_count", None)
                    usage["thinking_tokens"] = getattr(usage_metadata, "thoughts_token_count", None)
            except Exception:
                pass
            return usage

    @retry(
        retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, DeadlineExceeded)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    async def generate_content(self, sample_id, user_prompt, system_prompt=None):
        try:
            config = self.config

            if system_prompt and system_prompt.strip():
                config["system_instruction"] = system_prompt.strip()
            
            config = genai_types.GenerateContentConfig(**config)

            def _call():
                if config is not None:
                    return self.client.models.generate_content(
                        model=self.model_name,
                        contents=user_prompt.strip(),
                        config=config,
                    )
                else:
                    return self.client.models.generate_content(
                        model=self.model_name,
                        contents=user_prompt.strip(),
                    )

            response = await asyncio.to_thread(_call)

            text, thoughts = self._extract_text_and_thoughts_from_response(response)
            exception = self._extract_exception_from_response(response)
            usage = self._extract_usage_from_response(response)

            return ModelResponse(sample_id=sample_id, text=text, thoughts=thoughts, usage=usage, exception=exception)

        except Exception as e:
            return ModelResponse(sample_id=sample_id, text="", thoughts=None, usage=None, exception=str(e))

    async def generate(self, batch):
        tasks = []
        for sample in batch:
            user_prompt = sample["user_prompt"]
            system_prompt = sample.get("system_prompt")
            sample_id = sample["id"]
            tasks.append(asyncio.create_task(self.generate_content(sample_id, user_prompt, system_prompt=system_prompt)))
        return await asyncio.gather(*tasks)

def get_inference_service(model_name, api_key=None, model_args=None):
    if model_name.startswith("gemini"):
        return GoogleService(model_name, api_key, model_args=model_args)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

async def main():
    load_dotenv() 

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, help="Path to inference data in json", required=True)
    parser.add_argument("-a", "--api-key", type=str, help="Model API Key")
    parser.add_argument("-m", "--model", type=str, help="Model to use for inference")
    parser.add_argument("-t", "--temperature", type=float, help="Temperature for generation", default=1.0)
    parser.add_argument("-g", "--max-tokens", type=none_or_int, help="Max tokens for generation", default=None)
    parser.add_argument("-p", "--top-p", type=float, help="Top-p for generation", default=None)
    parser.add_argument("-k", "--top-k", type=none_or_int, help="Top-k for generation", default=None)
    parser.add_argument("-fp", "--frequency-penalty", type=float, help="Frequency penalty for generation", default=0)
    parser.add_argument("-pp", "--presence-penalty", type=float, help="Presence penalty for generation", default=0)
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory for inference results", default="outputs")
    parser.add_argument("-n", "--num-samples", type=int, help="Number of samples to run inference on", default=0)
    parser.add_argument("-b", "--batch-size", type=int, help="Batch size for inference", default=16)
    parser.add_argument("-r", "--resume-from-path", type=str, help="Resume inference from this path")
    parser.add_argument("-s", "--stop", type=none_or_str, help="Stop token for generation", default=None)
    parser.add_argument("--enable-thinking-summary", action="store_true", help="Whether to enable thinking summary in reasoning models")
    parser.add_argument("--enable-thinking-effort", type=none_or_str, choices=["minimal", "low", "medium", "high"], default=None, help="Effort level for thinking in reasoning models")
    
    args = parser.parse_args()
        
    input_data = read_json(args.datapath)
    data = input_data["data"]

    if args.resume_from_path:
        print("Resuming...")
        outputs = read_json(args.resume_from_path)
        outputs["data"] = [s for s in outputs["data"] if s["output"] and not s["exception"]]
        if args.max_tokens is not None:
            outputs["metadata"]["model_args"]["max_tokens"] = args.max_tokens
        resume_from_data_ids = [s["id"] for s in outputs["data"]]
        data = [s for s in data if s["id"] not in resume_from_data_ids]
        print(f"Found {len(data)} unfinished samples.")
        output_path = args.resume_from_path
    else:
        print("Starting a fresh run...")
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        datapath = pathlib.Path(args.datapath)
        model_name = args.model.split("/")[-1].lower()
        stop_token = args.stop.replace("\\n", "\n") if args.stop else None
        datetime_id = generate_datetime_id()
        output_path = os.path.join(args.output_dir, f"{datapath.stem}_{model_name}_{datetime_id}.json")

        outputs = {
            "metadata": {
                "source": args.datapath,
                "output_path": output_path,
                "size": len(data),
                "model": args.model,
                "model_name": model_name,
                "batch_size": args.batch_size,
                "num_samples": args.num_samples,
                "model_args": {
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "frequency_penalty": args.frequency_penalty,
                    "presence_penalty": args.presence_penalty,
                    "stop": stop_token,
                    "enable_thinking_summary": args.enable_thinking_summary,
                    "enable_thinking_effort": args.enable_thinking_effort,
                }
            },
            "metrics": {},
            "data": [],
        }

    if args.num_samples > 0:
        print(f"Limiting to {args.num_samples} samples...")
        data = data[: args.num_samples]

    print(f"Writing to {output_path}")

    # check all data have sample ids
    for sample in data:
        if "id" not in sample:
            raise ValueError(f"Sample {sample} does not have an id")

    service = get_inference_service(args.model, args.api_key, model_args=outputs["metadata"]["model_args"])

    for batch in tqdm(batched(data, size=args.batch_size),
                      total=math.ceil(len(data) / args.batch_size)):
        responses = await service.generate(batch)

        for response in responses:
            sample = {
                "id": response.sample_id,
                "output": response.text,
                "thoughts": response.thoughts,
                "usage": response.usage,
                "exception": str(response.exception) if response.exception else None,
                "result_id": generate_unique_id()
            }
            outputs["data"].append(sample)

        write_json(outputs, output_path)

    write_json(outputs, output_path)

if __name__ == "__main__":
    asyncio.run(main())