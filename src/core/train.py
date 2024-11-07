from trl import SFTTrainer
from datasets import DatasetDict
from transformers import TrainingArguments
from transformers import PreTrainedModel, PreTrainedTokenizer

from models.AppConfig import AppConfig

def train_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset: DatasetDict, config: AppConfig):
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        optim=config.training.optimizer,
        num_train_epochs=config.training.epochs,
        evaluation_strategy=config.training.evaluation_strategy,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        warmup_steps=config.training.warmup_steps,
        learning_rate=config.training.learning_rate,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        group_by_length=config.training.group_by_length,
        report_to=config.training.report_to
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        max_seq_length=512,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    trainer.train()
    return trainer