from abc import ABC, abstractmethod
import logging

class LMGenerator(ABC):
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)

    def __init__(self, model_path, log_file):
        self.model_id = model_path
        fh = logging.FileHandler(filename=log_file)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)

        self.LOGGER.addHandler(fh)
    
    @abstractmethod
    def __call__(self, prompt: str | list[str], sample_size=1, prefix=None) -> str | list[str]:
        pass
