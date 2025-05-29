from .base import LMGenerator
from .peft_generator import PeftGenerator
from .vllm_generator import VLLMGenerator
from .gpt_generator import GPTGenerator
#from .transformers_generator import TransformersGenerator

def make_generator(model_config):
    match model_config["loader"]:
        case "peft":
            generator = PeftGenerator(
                model_config["model_path"],
                model_config["log_file"],
                model_config["lora_path"],
            )
        case "vllm":
            generator = VLLMGenerator(
                model_config["model_path"], model_config["log_file"]
            )
        case "tonggpt":
            generator = GPTGenerator(
                model_config["model_path"], model_config["endpoint"], model_config.get("api_key"), model_config["log_file"]
            )
        case _:
            raise ValueError(f"Unknown loader {model_config['loader']}")
    return generator
