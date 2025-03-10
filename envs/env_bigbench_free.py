import re
import string
from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from .base_env import Env
import json
from collections import Counter



class BigbenchfreeEnv(Env):
    def __init__(self, ground_truth, is_react=True):
        super().__init__(
            ground_truth,
            invalid_hint="Invalid Action. Valid Action inputs are Lookup[<topic>] Search[<topic>] and Finish[<answer>] without any additional explanations.",
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

    def get_observation(self, action_type, argument,is_free_text=True):
        result = super().get_observation(action_type, argument,is_free_text=is_free_text)

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
        
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        if not isinstance(s, str):
            print(f"{s} is not a string")
            return ""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(s):
        if not s: 
            return []
        return BigbenchfreeEnv.normalize_answer(s).split()

    def exact_match_score(self, candidate_answer):
        """get exact match score."""
        print(f"{candidate_answer}:candidate_answer")
        if isinstance(self.ground_truth, str):
            correct_answer = self.ground_truth
        else:
            
            correct_answer = self.ground_truth.get('answer', '')
        print(f"{correct_answer}:correct_answer")
        return (BigbenchfreeEnv.normalize_answer(correct_answer) == BigbenchfreeEnv.normalize_answer(candidate_answer))


    def f1_score(self, candidate_answer):
        """计算F1分数."""
        print(f"{candidate_answer}:candidate_answer")
        if isinstance(self.ground_truth, str):
            correct_answer = self.ground_truth
        else:
            
            correct_answer = self.ground_truth.get('answer', '')
        print(f"{correct_answer}:correct_answer")
        gold_toks = BigbenchfreeEnv.get_tokens(correct_answer)
        pred_toks = BigbenchfreeEnv.get_tokens(candidate_answer)
        common = Counter(gold_toks) & Counter(pred_toks)  
        num_same = sum(common.values())
        
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            ## If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def is_correct(self, candidate_answer) -> bool:
        
        print(f"{candidate_answer}:candidate_answer")
        if isinstance(self.ground_truth, str):
            correct_answer = self.ground_truth
        else:
            
            correct_answer = self.ground_truth.get('answer', '')
        print(f"{correct_answer}:correct_answer")
        
        if BigbenchfreeEnv.normalize_answer(correct_answer) == BigbenchfreeEnv.normalize_answer(candidate_answer):
            self._is_correct = True
        else:
            self._is_correct = False

        return self._is_correct

