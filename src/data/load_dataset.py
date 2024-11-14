import multiprocessing
import datasets
from transformers import PreTrainedTokenizer

from ..data_models import AppConfig

def load_dataset(config: AppConfig, tokenizer: PreTrainedTokenizer) -> datasets.DatasetDict:
    dataset_config = config.dataset
    dataset = datasets.load_dataset(dataset_config.name, split="all")
    dataset = dataset.shuffle(seed=dataset_config.shuffle_seed).select(range(dataset_config.select_top_n))

    # TODO: Make sure this is the correct way to format the chat template
    def format_chat_template(row):
        row_json = [{"role": "user", "content": row["Patient"]},
                    {"role": "assistant", "content": row["Doctor"]}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row
    
    num_proc = multiprocessing.cpu_count()
    dataset = dataset.map(format_chat_template, num_proc=num_proc)
    dataset = dataset.train_test_split(test_size=dataset_config.test_split)
    return dataset