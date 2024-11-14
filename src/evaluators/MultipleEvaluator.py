import nltk
import evaluate
import numpy as np
from transformers import PreTrainedTokenizer

from . import BaseEvaluator
from ..enums import EvaluatorInputType

class MultipleEvaluator(BaseEvaluator):
    def __init__(self, tokenizer: PreTrainedTokenizer, eval_metrics: list[str]):
        super().__init__(tokenizer)
        self.eval_metrics_str = evaluate.combine([evaluate.load(metric) for metric in eval_metrics if isinstance(EvaluatorInputType[metric.upper()].value, str)])
        self.eval_metrics_int = evaluate.combine([evaluate.load(metric) for metric in eval_metrics if isinstance(EvaluatorInputType[metric.upper()].value, int)])

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        # Get the token integer (index) of the maximum logit
        predictions = np.argmax(logits, axis=-1)

        # # Replace padding tokens with -100 so tokenizer decodes properly
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        # Decode predictions and labels from integers to strings
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        # This shoud not affect the other metrics since its applied to both predictions and labels
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result_int = self.eval_metrics_int.compute(predictions=predictions, references=labels)
        result_str = self.eval_metrics_str.compute(predictions=decoded_preds, references=decoded_labels)

        result = result_str | result_int
        return result