from trl import SFTTrainer
from datasets import DatasetDict
from transformers import TrainingArguments
from transformers import PreTrainedModel, PreTrainedTokenizer

from models.AppConfig import AppConfig
# TODO: Create EvaluatorFactory
from evaluators.MultipleEvaluator import MultipleEvaluator

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
        report_to=config.training.report_to,
        do_predict=True
    )

    evaluator = MultipleEvaluator(tokenizer, config.training.evaluation_metrics)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # TODO: Remove this hardcoded efficency
        max_seq_length=512,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        compute_metrics=evaluator.compute_metrics,
    )
    trainer.train()
    return trainer