import os
from llms.base import LMGenerator
from openai import AzureOpenAI

class GPTGenerator(LMGenerator):
    def __init__(self, model_path, endpoint, api_key, log_file='gpt_generate.log'):
        super().__init__(model_path, log_file)
        if api_key is None:
            api_key=os.getenv("OPENAI_KEY")
        else:
            raise ValueError("Use OPENAI_KEY env variable")
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint,
        )

    def __call__(self, prompt, sample_size=1, prefix=None):
        assert sample_size == 1
        assert prefix is None
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": prompt}
            ]
        elif isinstance(prompt, list):
            messages = [
                {"role": "system", "content": prompt[0]}
            ]
            for p in prompt[1:]:
                messages.append({"role": "user", "content": p})
        else:
            raise ValueError(f"Unknown prompt type {type(prompt)}")
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content