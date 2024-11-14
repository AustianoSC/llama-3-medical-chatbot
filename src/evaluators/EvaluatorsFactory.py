from typing import Union
from transformers import PreTrainedTokenizer

from . import BaseEvaluator, F1Evaluator, ExactMatchEvaluator, BleuEvaluator, RougeEvaluator, MultipleEvaluator

from ..enums import Evaluators

class EvaluatorsFactory:
    _evaluator_map = {
        Evaluators.EXACT_MATCH.value: ExactMatchEvaluator,
        Evaluators.F1.value: F1Evaluator,
        Evaluators.BLEU.value: BleuEvaluator,
        Evaluators.ROUGE.value: RougeEvaluator
    }

    @staticmethod
    def get_evaluator(evaluation_metrics: Union[list[str], str], tokenizer: PreTrainedTokenizer) -> BaseEvaluator:
        if isinstance(evaluation_metrics, str):
            return EvaluatorsFactory._evaluator_map[evaluation_metrics.lower()](tokenizer)
        else:
            evaluation_metrics = [evaluator.lower() for evaluator in evaluation_metrics]
            return MultipleEvaluator(tokenizer, evaluation_metrics)