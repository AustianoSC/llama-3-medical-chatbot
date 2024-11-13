import evaluate
import numpy as np
from transformers import PreTrainedTokenizer

from .BaseEvaluator import BaseEvaluator

class F1Evaluator(BaseEvaluator):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer)
        self.f1 = evaluate.load("f1")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        # Get the token integer (index) of the maximum logit
        predictions = np.argmax(logits, axis=-1)

        # Decode predictions and labels from integers to strings
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.f1.compute(predictions=decoded_preds, references=decoded_labels)
        return result