from transformers import PreTrainedTokenizer

from ...enums import DatasetProcessors
from .BaseProcessor import BaseProcessor
from .AiMedicalChatbotProcessor import AiMedicalChatbotProcessor

class PreProcessorFactory:
    _preprocessor_map = {
        DatasetProcessors.RUSLANMV_AI_MEDICAL_CHATBOT.name: AiMedicalChatbotProcessor,
    }

    @staticmethod
    def get_preprocessor(dataset_name: str, tokenizer: PreTrainedTokenizer) -> BaseProcessor:
        return PreProcessorFactory._preprocessor_map[dataset_name.upper()](tokenizer)