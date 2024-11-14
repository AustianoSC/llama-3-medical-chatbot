import evaluate
import numpy as np
from transformers import PreTrainedTokenizer

from . import BaseEvaluator

class F1Evaluator(BaseEvaluator):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer)
        self.f1 = evaluate.load("f1")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        # Get the token integer (index) of the maximum logit
        predictions = np.argmax(logits, axis=-1)

        result = self.f1.compute(predictions=predictions, references=labels)
        return result