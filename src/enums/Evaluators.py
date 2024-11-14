from enum import Enum

class Evaluators(Enum):
    BLEU = 'bleu'
    EXACT_MATCH = 'exact_match'
    F1 = 'f1'
    ROUGE = 'rouge'

class EvaluatorInputType(Enum):
    BLEU = ""
    EXACT_MATCH = ""
    F1 = 0
    ROUGE = ""