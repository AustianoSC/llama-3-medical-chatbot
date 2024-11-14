import multiprocessing
import datasets
from transformers import PreTrainedTokenizer

from ..data_models import DatasetConfig

def load_dataset(config: DatasetConfig, tokenizer: PreTrainedTokenizer) -> datasets.DatasetDict:
    dataset = datasets.load_dataset(config.name, split="all")
    dataset = dataset.shuffle(seed=config.shuffle_seed)

    if config.select_top_n:
        dataset = dataset.select(range(config.select_top_n))

    def format_chat_template(row):
        row_json = [{"role": "user", "content": row["Patient"]},
                    {"role": "assistant", "content": row["Doctor"]}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row
    
    num_proc = multiprocessing.cpu_count()
    dataset = dataset.map(format_chat_template, num_proc=num_proc)
    dataset = dataset.train_test_split(test_size=config.test_split)
    return dataset