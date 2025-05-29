from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

from llms.base import LMGenerator


class PeftGenerator(LMGenerator):
    def __init__(self, model_path, log_file, lora_path):
        super().__init__(f'{model_path}::{lora_path}', log_file)
        print("!!! PeftGenerator init !!!")
        self.LOGGER.info('Load Peft model')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda',torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, model_id=lora_path)
        #model = PeftModel.from_pretrained(model, model_id="/scratch/nlp/lijiaqi/RFL/results/checkpoint-500/")
        self.model = model.merge_and_unload()
        self.LOGGER.info("Peft model loaded.")

    def __call__(self, prompt, sample_size=1, prefix=None):
        assert sample_size==1
        if isinstance(prompt, str):
            packed= f"<|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            packed= f"<|start_header_id|>system<|end_header_id|>\n\n{prompt[0]}<|eot_id|>"
            for p in prompt[1:]:
                packed += f"<|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|>"
            packed += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        if prefix is not None:
            packed = packed + prefix
        model_inputs = self.tokenizer([packed], add_special_tokens=False, return_tensors="pt").to('cuda')
            
        generated_ids = self.model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=500,  # !!!!!!!
                        do_sample=False,
                        temperature=0.0, 
                        repetition_penalty=1.2,
                        length_penalty=1.0,
                        eos_token_id=self.tokenizer.encode('<|eot_id|>')[0],
                        pad_token_id=self.tokenizer.eos_token_id
                    )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
