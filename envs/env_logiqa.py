import re
import string
from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from .base_env import Env


class LogiQAEnv(Env):
    def __init__(self, ground_truth, is_react=True):
        super().__init__(
            ground_truth,
            invalid_hint="Invalid Action. Valid Action inputs are Lookup[<topic>] Search[<topic>] and Finish[<answer>] without any addional explanations.",
        )
        self.is_react = is_react
        if is_react:
            self.docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia())
        self._cached_answer = ""
        self._is_correct = False

    def parse_action(self, string):
        match = True
        if 'Finish[' in string:
            action_type = 'Finish'
            argument = string.split("Finish[")[1].split("]")[0]
        elif 'Search[' in string:
            action_type = 'Search'
            argument = string.split("Search[")[1].split("]")[0]
        elif 'Lookup[' in string:
            action_type = 'Lookup'
            argument = string.split("Lookup[")[1].split("]")[0]
        else:
            match = False

        if match:
            return action_type, argument
        else:
            return None

    def get_observation(self, action_type, argument,is_free_text=False):
        result = super().get_observation(action_type, argument)

        def format_step(step: str) -> str:
            return step.strip("\n").strip().replace("\n", "")

        if result is not None:  # i.e., is general action
            return result
        if self.is_react:
            if action_type == "Search":
                try:
                    return False, format_step(self.docstore.search(argument)), None
                except Exception as e:
                    print(e)
                    return (
                        False,
                        f'Could not find that page "{argument}", please try again.',
                        None,
                    )

            elif action_type == "Lookup":
                try:
                    return False, format_step(self.docstore.lookup(argument)), None
                except ValueError:
                    return (
                        False,
                        "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.",
                        None,
                    )

        return None

    def is_correct(self, candidate_answer) -> bool:
        if candidate_answer != self._cached_answer:
            self._cached_answer = candidate_answer
            self._is_correct = self.ground_truth.lower() == candidate_answer.lower()

        return self._is_correct
