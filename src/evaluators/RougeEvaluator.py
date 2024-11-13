import nltk
import evaluate
import numpy as np
from transformers import PreTrainedTokenizer

from .BaseEvaluator import BaseEvaluator

class RougeEvaluator(BaseEvaluator):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer)
        self.rouge = evaluate.load("rouge")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        # Get the token integer (index) of the maximum logit
        predictions = np.argmax(logits, axis=-1)

        # Decode predictions and labels from integers to strings
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return result