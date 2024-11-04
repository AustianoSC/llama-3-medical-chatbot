from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

from models.AppConfig import AppConfig

def prepare_dataset(config: AppConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    dataset_config = config['dataset']
    dataset = load_dataset(dataset_config['name'], split="all")
    dataset = dataset.shuffle(seed=dataset_config['shuffle_seed']).select(range(dataset_config['select_top_n']))

    def format_chat_template(row):
        row_json = [{"role": "user", "content": row["Patient"]},
                    {"role": "assistant", "content": row["Doctor"]}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = dataset.map(format_chat_template, num_proc=4)
    dataset = dataset.train_test_split(test_size=dataset_config['test_size'])
    return dataset