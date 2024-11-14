from enum import Enum

class Evaluators(Enum):
    BLEU = 'bleu'
    ExactMatch = 'exact_match'
    F1 = 'f1'
    ROUGE = 'rouge'