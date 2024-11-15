from transformers import PreTrainedTokenizer

from ...enums import DatasetProcessors
from .BaseProcessor import BaseProcessor

class PreProcessorFactory:
    @staticmethod
    def get_preprocessor(dataset_name: str, tokenizer: PreTrainedTokenizer) -> BaseProcessor:
        return DatasetProcessors[dataset_name.upper()].value(tokenizer)