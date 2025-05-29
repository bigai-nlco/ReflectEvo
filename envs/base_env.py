from abc import ABC, abstractmethod


class Env(ABC):
    def __init__(
        self,
        ground_truth,
        invalid_hint,
    ) -> None:
        super().__init__()
        self.ground_truth = ground_truth
        self.finished = False
        self.invalid_hint = invalid_hint

    @abstractmethod
    def parse_action(self, string):
        """
        Parses the given string and extracts the action type and argument.

        Args:
            string (str): The string to be parsed.

        Returns:
            Tuple[str, str] or None: A tuple containing the action type and argument if the string is successfully parsed.
            Otherwise, returns None.

        """

    def get_observation(self, action_type, argument,is_free_text=False):
        """
        Generates an observation based on the given action type and argument.

        Args:
            action_type (str): The type of action to perform.
            argument (str): The argument for the action.

        Returns:
            None if the action is invalid.
            Otherwise a triple of (is_finish, obs, answer):
            - bool (is_finish): Whether to finish the environment.
            - str (obs): The generated observation.
            - optional[str] (answer): The extracted answer for evaluation (only if is_finish is True, otherwise None).

        Raises:
            None

        Notes:
            - If the action type is 'Finish', the answer is set to the argument.
                - If the answer is correct, the observation is set to 'Answer is CORRECT'.
                - If the answer is incorrect, the observation is set to 'Answer is INCORRECT'.
        """
        if not is_free_text:
            observation = ""
            if action_type == "Finish":
                if self.is_correct(argument):
                    observation = "Answer is CORRECT"
                else:
                    observation = "Answer is INCORRECT"
                self.finished = True
                return self.finished, observation, argument
            else:
                return None
        else:
            observation = ""
            if action_type == "Finish":
                print(f"Answer: {argument}")
                f1_score=self.f1_score(argument)
                exact_match_score=self.exact_match_score(argument)
                if exact_match_score == 1.0:
                    observation = "Answer is CORRECT"
                else:
                    observation = f'Answer is INCORRECT. F1 Score: {f1_score} Exact Match Score: {exact_match_score}'
                self.finished = True
                return self.finished, observation, argument


    @abstractmethod
    def is_correct(self, candidate_answer) -> bool:
        """
        Check if the candidate answer is correct.

        Args:
            candidate_answer: The candidate answer to be checked.

        Returns:
            bool: True if the candidate answer is correct, False otherwise.
        """

    def reset(self):
        self.finished = False

    def f1_score(self, candidate_answer):
        """
        Calculate the F1 score.

        Args:
            candidate_answer: The candidate answer to be evaluated.

        Returns:
            float: The F1 score.
        """
    def exact_match_score(self, candidate_answer):
        """
        Calculate the exact match score.

        Args:
            candidate_answer: The candidate answer to be evaluated.

        Returns:
            float: The exact match score.
        """    
