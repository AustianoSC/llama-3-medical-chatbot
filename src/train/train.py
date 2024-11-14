from trl import SFTTrainer
from datasets import DatasetDict
from transformers import TrainingArguments
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..data_models import TrainingConfig
from ..evaluators import EvaluatorsFactory

def train_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset: DatasetDict, training_config: TrainingConfig) -> SFTTrainer:
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        optim=training_config.optimizer,
        num_train_epochs=training_config.epochs,
        evaluation_strategy=training_config.evaluation_strategy,
        eval_steps=training_config.eval_steps,
        logging_steps=training_config.logging_steps,
        warmup_steps=training_config.warmup_steps,
        learning_rate=training_config.learning_rate,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        group_by_length=training_config.group_by_length,
        report_to=training_config.report_to,
        do_predict=True
    )

    evaluator = EvaluatorsFactory.get_evaluator(training_config.evaluation_metrics, tokenizer)

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