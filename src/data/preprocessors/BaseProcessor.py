from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer

class BaseProcessor(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def format_chat_template(self, row):
        pass