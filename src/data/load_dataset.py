import multiprocessing
import datasets
from transformers import PreTrainedTokenizer

from .preprocessors import PreProcessorFactory
from ..data_models import DatasetConfig

def load_dataset(config: DatasetConfig, tokenizer: PreTrainedTokenizer) -> datasets.DatasetDict:
    dataset = datasets.load_dataset(config.name, split="all")
    dataset = dataset.shuffle(seed=config.shuffle_seed)

    if config.select_top_n:
        dataset = dataset.select(range(config.select_top_n))

    preprocessor = PreProcessorFactory.get_preprocessor(config.name.replace('/', '_'), tokenizer)
    
    num_proc = multiprocessing.cpu_count()
    dataset = dataset.map(preprocessor.format_chat_template, num_proc=num_proc)
    dataset = dataset.train_test_split(test_size=config.test_split)
    return dataset