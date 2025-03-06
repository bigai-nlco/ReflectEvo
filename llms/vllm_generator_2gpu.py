import math
from vllm import LLM, RequestOutput, SamplingParams
import pynvml
import torch

from llms.base import LMGenerator


class VLLMGenerator_2gpu(LMGenerator):
    def __init__(self, model_path, log_file="vllm_generate.log"):
        self.model_name = ''
        match model_path:
            case '/home/lijiaqi/PLMs/Meta-Llama-3-8B-Instruct': self.model_name = 'llama'
            case '/home/lijiaqi/PLMs/gemma-2-9b-it': self.model_name = 'gemma'
            case '/home/lijiaqi/PLMs/Mistral-7B-Instruct-v0.2': self.model_name = 'mistral'
        super().__init__(model_path, log_file)
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            print(f"GPU 0: {gpu_name}")
            if "3090" in gpu_name:
                self.llm = LLM(model=model_path, gpu_memory_utilization=0.95)
                self.top_p = 0.9
                self.top_k = -1
            if "4090" in gpu_name:
                self.llm = LLM(model=model_path, gpu_memory_utilization=0.95)
                self.top_p = 0.9
                self.top_k = -1
            if ("A800" in gpu_name) or ("A100" in gpu_name):
                # 这里改一下，llama占0.3就行，其他的占0.3不够       咋又行了
                self.llm = LLM(model=model_path, gpu_memory_utilization=0.3)
                '''if self.model_name == 'llama':
                    self.llm = LLM(model=model_path, gpu_memory_utilization=0.3)
                else: 
                    self.llm = LLM(model=model_path, gpu_memory_utilization=0.3)'''
                self.top_p = 1.0
                self.top_k = -1

        except pynvml.NVMLError as err:
            print(f"Failed to initialize NVML: {err}")
            exit(1)

    def __call__(self, prompt, sample_size=1, prefix=None):
        tokenizer = self.llm.get_tokenizer()
        if sample_size == 1:
            temperature = 0
        else:
            temperature = 0.8
        sampling_params = SamplingParams(
            n=sample_size,
            # best_of=sample_size,
            temperature=temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=512,         # 这里原本是200！！！
            stop_token_ids=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )
        if isinstance(prompt, str):
            if self.model_name == 'llama':      # 这里也要改。prob函数里可能也要改  现在都改成user了
                input_tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                input_tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        elif isinstance(prompt, list):
            assert len(prompt) > 1
            '''if self.model_name == 'llama':          # 除了llama外，其他llm都不支持sys_str
                input_tokens = tokenizer.apply_chat_template(
                    [{"role": "system", "content": prompt[0]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )'''
            input_tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt[0]}],
                    tokenize=False,
                    add_generation_prompt=True,
            )
            for p in prompt[1:-1]:
                input_tokens += tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            input_tokens += tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt[-1]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        if prefix is not None:
            input_tokens += prefix
        input_device = torch.device("cuda:1")  # 模型所在的设备
        output_device = torch.device("cuda:0")  # 调用方希望接收结果的设备
        if is_refl:
            input_tokens = torch.tensor(input_tokens).to(input_device)
        output: RequestOutput = self.llm.generate([input_tokens], sampling_params)[0]
        if is_refl:
            output = output.to(output_device)
        self.LOGGER.debug("===Prompt LLM===")
        for i, data in enumerate(output.outputs):
            self.LOGGER.debug(
                "Output %d:\n\t>>> %s\n\n\t>>> %s",
                i,
                output.prompt,
                data.text,
            )
        self.LOGGER.debug("===End Prompt LLM===")
        if sample_size == 1:
            return output.outputs[0].text
        else:
            res = [data.text for data in output.outputs]
            return res

    def prob(self, prompt, prefix):
        tokenizer = self.llm.get_tokenizer()
        input_str = tokenizer.apply_chat_template(  
            [{"role": "user", "content": prompt}],        # 这里也改成user吧
            tokenize=False,
            add_generation_prompt=True,
        )
        start_pos = len(
            tokenizer(input_str, return_tensors="pt")["input_ids"].squeeze().tolist()
        )
        input_str += prefix
        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            max_tokens=1,
            top_p=self.top_p,
            top_k=self.top_k,
            stop_token_ids=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            prompt_logprobs=1,
        )
        output: RequestOutput = self.llm.generate([input_str], sampling_params)[0]

        result = dict()

        # All output

        logprobs = output.prompt_logprobs[start_pos:]
        prefix_token_ids = (
            tokenizer(prefix, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ]
            .squeeze()
            .tolist()
        )
        # decoded_prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_token_ids)
        prefix_logprobs = []
        for token_id, token_logprobs in zip(prefix_token_ids, logprobs):
            logprob = token_logprobs[token_id].logprob
            prefix_logprobs.append(logprob)

        log_prob_sum = sum(prefix_logprobs)
        probability = math.exp(log_prob_sum)

        result["ground_truth_all"] = probability
        log_prob_sum = sum(prefix_logprobs) / len(prefix_logprobs)
        probability = math.exp(log_prob_sum)
        result["norm_ground_truth_all"] = probability
        # result["string_all"] = prefix_string

        # Fill-in ouptut

        # HACK remove ['Action', ':', 'Finish']
        logprobs = output.prompt_logprobs[start_pos + 3 :]
        prefix_token_ids = (
            tokenizer(prefix, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ]
            .squeeze()
            .tolist()[3:]
        )
        # remove ['[', ']'] if possible
        if tokenizer.convert_ids_to_tokens(prefix_token_ids[0]) == '[':
            prefix_token_ids = prefix_token_ids[1:]
            logprobs = logprobs[1:]
        if tokenizer.convert_ids_to_tokens(prefix_token_ids[-1]) == ']':
            prefix_token_ids = prefix_token_ids[:-1]
            logprobs = logprobs[:-1]
        # decoded_prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_token_ids)
        # # prefix_string = tokenizer.convert_tokens_to_string(decoded_prefix_tokens)
        # # print(prefix_string)
        # # print(decoded_prefix_tokens)
        # prefix_logprobs = []
        # for token_id, token_logprobs in zip(decoded_prefix_tokens, logprobs):
        #     # Search for the logprob corresponding to the sampled token (token_id)
        #     for logprob in token_logprobs.values():
        #         if token_id.startswith("Ġ"):  # HACK
        #             temp_token = " " + token_id[1:]
        #         else:
        #             temp_token = token_id
        #         if logprob.decoded_token == temp_token:
        #             prefix_logprobs.append(logprob.logprob)
        #             self.LOGGER.debug(
        #                 "LOGPROB of fill in: %s: %s",
        #                 logprob.decoded_token,
        #                 logprob.logprob,
        #             )
        #             break
        #         else:
        #             # Handle the case where the token_id is not found, which might happen in rare cases.
        #             raise ValueError(
        #                 f"Token ID {temp_token} not found in logprobs: {token_logprobs}"
        #             )
        for token_id, token_logprobs in zip(prefix_token_ids, logprobs):
            logprob = token_logprobs[token_id].logprob
            prefix_logprobs.append(logprob)
            # else:
            #     raise ValueError(f"Token ID {token_id} not found in logprobs: {token_logprobs}")

        log_prob_sum = sum(prefix_logprobs)
        probability = math.exp(log_prob_sum)

        result["ground_truth_fill_in"] = probability
        log_prob_sum = sum(prefix_logprobs) / len(prefix_logprobs)
        probability = math.exp(log_prob_sum)
        result["norm_ground_truth_fill_in"] = probability
        # result["string_fill_in"] = prefix_string

        return result
