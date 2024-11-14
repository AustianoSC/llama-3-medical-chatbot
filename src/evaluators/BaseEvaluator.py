from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer

class BaseEvaluator(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_metrics(self, eval_pred):
        pass