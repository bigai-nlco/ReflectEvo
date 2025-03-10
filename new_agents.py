from abc import ABC
from dataclasses import dataclass
import re
from typing import List
from langchain.prompts import PromptTemplate
import tiktoken
from prompts.prompts import (
    REFLECTION_HEADER,
)
from envs.base_env import Env
from envs.env_mbpp import MBPPEnv
from llms import VLLMGenerator

def format_step(step: str) -> str:
    return step



def format_reflections(reflections: List[str], scratchpad: str, header: str = REFLECTION_HEADER, setting: str=0, trail = 0) -> str:
    if trail == 1:
        return ""
    elif trail ==2:
        if setting == 1:
            print("not using scratchpad")
            if reflections == []:
                return "You were unsuccessful in answering the question."
            else:
                reflections_str = (
                    header + "Reflections:\n- " + "\n- ".join([r.strip() for r in reflections])
                )
                reflections_str += "\n\nYou were unsuccessful in answering the question."
                return reflections_str
        elif setting == 2:
            print("using scratchpad")
            if reflections == []:
                reflections_str = "You were unsuccessful in answering the question.\nBelow is your Previous trial and your incorrect solution:"+  "\n"+ scratchpad
                return reflections_str
            else:
                reflections_str = (
                    header + "Reflections:\n- " + "\n- ".join([r.strip() for r in reflections])
                )
                reflections_str += "\n\nYou were unsuccessful in answering the question.\nBelow is your Previous trial and your incorrect solution:"+ "\n"+ scratchpad
                return reflections_str
        else:
            print("setting error")

@dataclass
class BatchReactReflectAgent(ABC):
    question: str
    answer: str
    reason_llm: VLLMGenerator
    reflect_llm: VLLMGenerator
    env: Env
    agent_prompt: PromptTemplate
    reflect_prompt: tuple[str, PromptTemplate]
    examples: str
    demand: str
    max_steps: int = 5
    max_sample: int = 5
    max_retry: int = 1

    def __post_init__(self):

        self.enc = tiktoken.encoding_for_model("text-davinci-003")
        self.generated_answer = ""
        self.step_n = 1
        self.reflections: List[str] = []
        self.reflections_str = ""
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ""
        self.generated_answer = ""
        self.retry_cnt = 0
        self.env.reset()

    def reset(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ""
        self.generated_answer = ""
        self.retry_cnt = 0
        self.env.reset()

    def is_correct(self) -> bool:
        return self.env.is_correct(self.generated_answer)

    def is_halted(self) -> bool:
        return (
            (self.step_n > self.max_steps)
            or (len(self.enc.encode(self._build_agent_prompt())) > 3896)
        ) and not self.finished

    def prompt_reflection(self, sample_size=None) -> str | list[str]:
        if sample_size is None:
            sample_size = self.max_sample
        if sample_size == 1:
            return format_step(
                self.reflect_llm(
                    [self.reflect_prompt[0], self._build_reflection_prompt()], 1
                )
            )
        else:
            return [
                format_step(s)
                for s in self.reflect_llm(
                    [self.reflect_prompt[0], self._build_reflection_prompt()],
                    sample_size,
                )
            ]

    def run(self, reset=True, reflection=None, setting=0, trail=0) -> None:
        if (self.finished or self.is_halted()) and not self.is_correct():
            if reflection is None:
                self.reflections = []
                self.reflections_str = format_reflections(self.reflections,self.scratchpad,setting=setting,trail=trail)
            else:
                self.reflections = [reflection]
                self.reflections_str = format_reflections(self.reflections,self.scratchpad,setting=setting,trail=trail)
        if not self.is_correct():
            if reset:
                self.reset()
            while not self.is_halted() and not self.finished:
                self.step()

    def step(self) -> None:

        self.scratchpad += f"\nThought {self.step_n}:"
        thought = self.prompt_agent(prefix=f"Thought {self.step_n}: ")
        if f"Thought {self.step_n}: " in thought:
            thought = thought.split(f"Thought {self.step_n}: ")[1]
        if f"Action {self.step_n}: " in thought:
            thought = thought.split(f"Action {self.step_n}: ")[0]
        self.scratchpad += " " + thought



        self.scratchpad += f"\nAction {self.step_n}:"
        action_type = ""
        argument = ""
        action = self.prompt_agent(prefix=f"Action {self.step_n}: ")
        if f"Action {self.step_n}:" in action:
            action = action.split(f"Action {self.step_n}:")[1]
        action = action.split("]")[0] + "]"
        try:
            action_type, argument = self.env.parse_action(action)
            print('argument:', argument)
        except TypeError:
            pass


        self.scratchpad += " " + action



        self.scratchpad += f"\nObservation {self.step_n}: "
        observation = self.env.get_observation(action_type, argument)
        print('observation:',observation)
        if observation is None:
            self.scratchpad += self.env.invalid_hint
        else:
            self.finished = observation[0]
            self.scratchpad += observation[1]
            if observation[0]:
                self.generated_answer = observation[2]

        self.step_n += 1


    def prompt_agent(self, prefix=None) -> str:
        return self.reason_llm(self._build_agent_prompt(), prefix=prefix)

    def _build_reflection_prompt(self) -> str:
        prompt = self.reflect_prompt
        if 'answer' in prompt:
            prompt = prompt.format(
                question=self.question, scratchpad=self.scratchpad, tokenizer=self, answer=self.answer, demand=self.demand
            )
        else:
            prompt = prompt.format(
                question=self.question, scratchpad=self.scratchpad, tokenizer=self, demand=self.demand
            )
            print(f"reflection prompt is:{prompt}")
        return prompt

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
            examples=self.examples,
            reflections=self.reflections_str,
            question=self.question,
            scratchpad=self.scratchpad,
            max_steps=self.max_steps,
        )


@dataclass
class BatchCOTReflectAgent:
    question: str
    answer: str
    reason_llm: VLLMGenerator
    reflect_llm: VLLMGenerator
    env: Env
    agent_prompt: PromptTemplate
    reflect_prompt: tuple[str, PromptTemplate]
    examples: str
    demand: str
    max_sample: int = 5
    max_retry: int = 1

    def __post_init__(self):


        self.generated_answer = ""
        self.reflections: List[str] = []
        self.reflections_str = ""
        self.step_n = 0
        self.finished = False
        self.scratchpad: str = ""
        self.generated_answer = ""
        self.env.reset()

        self.answer_prob = None

    def analyze_prob(self, temp_scratchpad):
        if "\nAction: " in temp_scratchpad:
            temp_scratchpad = temp_scratchpad.split("\nAction: ")[0] + "\nAction: "
        cached_scratchpad = self.scratchpad
        self.scratchpad = temp_scratchpad
        print(f"Action: Finish[{self.env.ground_truth}]")
        result = self.reason_llm.prob(
            self._build_agent_prompt(), f"Action: Finish[{self.env.ground_truth}]"
        )
        self.scratchpad = cached_scratchpad
        return result

    def reset(self) -> None:
        self.step_n = 0
        self.finished = False
        self.scratchpad: str = ""
        self.generated_answer = ""
        self.env.reset()

    def is_correct(self) -> bool:
        return self.env.is_correct(self.generated_answer)

    def run_c2(self, reset=True, reflection=None, setting=0, trail=0,is_free_text=False) -> None:
        if reflection is None:
            self.reflections = []
            self.reflections_str = format_reflections(self.reflections,self.scratchpad,setting=setting,trail=trail)
            self.reset()
        else:
            self.reflections = [reflection]
            self.reflections_str = format_reflections(self.reflections,self.scratchpad,setting=setting,trail=trail)
            self.reset()
        self.step_c2(is_free_text=is_free_text)
        self.step_n += 1
    def step_c2(self,is_free_text=False) -> None:
        self.scratchpad = ''
        action_type = ""
        argument = ""
        try:
            action_type, argument = self.env.parse_action(self.reflections[0])
        except TypeError:
            pass

        observation = self.env.get_observation(action_type, argument,is_free_text=is_free_text)
        print('observation:', observation)
        if observation is None:
            self.scratchpad += self.env.invalid_hint
        else:
            self.finished = observation[0]
            self.scratchpad += observation[1]
            self.generated_answer = observation[2]
    def run_SFT(self, reflection=None,setting=0,trail=0,is_free_text=False) -> None:
        if reflection is None:
            self.reflections = []
            self.reflections_str = format_reflections(self.reflections,self.scratchpad,setting=setting,trail=trail)

            self.reset()

        else:
            self.reflections = [reflection]
            self.reflections_str = format_reflections(self.reflections,self.scratchpad,setting=setting,trail=trail)

            self.reset()

        self.step_SFT()
        self.step_n += 1

    def step_SFT(self,is_free_text=False) -> None:

        thought_action = self.prompt_agent()
        print('原始model输出: ',thought_action)
        self.scratchpad += " " + thought_action

        action_type = ""
        argument = ""
        try:
            action_type, argument = self.env.parse_action(thought_action)
        except TypeError:
            pass

        self.scratchpad += "\nObservation: "

        observation = self.env.get_observation(action_type, argument,is_free_text=is_free_text)

        if observation is None:
            self.scratchpad += self.env.invalid_hint
        else:
            self.finished = observation[0]
            self.scratchpad += observation[1]
            self.generated_answer = observation[2]

    def run(self, reflection=None,setting=0,trail=0,is_free_text=False) -> None:
        if reflection is None:
            self.reflections = []
            self.reflections_str = format_reflections(self.reflections,self.scratchpad,setting=setting,trail=trail)

            self.reset()

        else:
            self.reflections = [reflection]
            self.reflections_str = format_reflections(self.reflections,self.scratchpad,setting=setting,trail=trail)

            self.reset()

        self.step(is_free_text=is_free_text)
        self.step_n += 1

    def step(self,is_free_text=False) -> None:

        print(f"step_is_free_text:{is_free_text}")
        self.scratchpad += "\nThought:"
        thought = self.prompt_agent(prefix=f"Thought: ")
        if "Thought:" in thought:
            thought = thought.split("Thought:")[1]
        if thought.find("Action:"):
            thought = thought.split("Action:")[0]
        if thought.find("[BEGIN]"):
            thought = thought.split("[BEGIN]")[0]
        self.scratchpad += " " + thought



        self.scratchpad += "\nAction:"
        action_type = ""
        argument = ""
        cnt = 0
        while cnt < self.max_retry:
            cnt += 1
            action = self.prompt_agent(prefix=f"Action: ")
            print("LLM Action输出：", action)

            try:
                action_type, argument = self.env.parse_action(action)
                if len(action_type) > 0:
                    break
            except TypeError:
                pass

        self.scratchpad += " " + action


        self.scratchpad += "\nObservation: "

        observation = self.env.get_observation(action_type, argument,is_free_text=is_free_text)

        if observation is None:
            self.scratchpad += self.env.invalid_hint
        else:
            self.finished = observation[0]
            self.scratchpad += observation[1]
            self.generated_answer = observation[2]

    def prompt_agent(self, prefix=None) -> str:
        return self.reason_llm(self._build_agent_prompt(), prefix=prefix)

    def prompt_reflection(self, sample_size=None,previous_answer=None, trail = 0) -> str | list[str]:
        if sample_size is None:
            sample_size = self.max_sample
        if sample_size == 1:
            a = self.reflect_llm(self._build_reflection_prompt(trail=trail,previous_answer=previous_answer), 1)
            return format_step(self.reflect_llm(self._build_reflection_prompt(trail=trail,previous_answer=previous_answer), 1))
        else:
            return [
                format_step(s)
                for s in self.reflect_llm(self._build_reflection_prompt(trail,previous_answer), sample_size)
            ]

    def _build_reflection_prompt(self,trail=0, previous_answer=None) -> str:
        prompt = self.reflect_prompt

        if 'answer' in prompt:
            if trail>=3:
                scratchpad = f"{previous_answer}\n{self.scratchpad}"
            else:
                scratchpad = self.scratchpad
            
            prompt = prompt.format(
                question=self.question, scratchpad=scratchpad, tokenizer=self, answer=self.answer, demand=self.demand
            )
        else:
            if trail>=3:

                scratchpad = f"{previous_answer}\n{self.scratchpad}"
            else:
                scratchpad = self.scratchpad
            
            prompt = prompt.format(
                question=self.question, scratchpad=scratchpad, tokenizer=self, demand=self.demand
            )
        print(f"trail:{trail},reflection prompt is:{prompt}")
        return prompt

    def _build_agent_prompt(self) -> str:

        return self.agent_prompt.format(
            examples=self.examples,
            reflections=self.reflections_str,
            question=self.question,
            scratchpad=self.scratchpad,
            )
