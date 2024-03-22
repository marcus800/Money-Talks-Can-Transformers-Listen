import os
from tqdm import tqdm
from transformers import TrainingArguments
from datasets import Dataset
import pandas as pd

from trl import DPOTrainer

from dataset import HF_Dataset

tqdm.pandas()

if __name__ == "__main__":

    HF_TOKEN = os.environ.get("HF_WRITE_TOKEN")
    assert HF_TOKEN is not None, "You must set the HF_WRITE_TOKEN environment variable."

    os.environ["WANDB_PROJECT"] = "MLP-LLM-DPO"
    os.environ["WANDB_LOG_MODEL"] = "MLP-LLM-dpo-7b"

    model_name_or_path = "seanmemery/MLP-FinLLM-7b-it"
    csv_location = "train_data/dpo_data.csv"
    dataset_text_field = "text"
    max_seq_length = 256
    repo_name = "seanmemery/MLP-FinLLM-dpo-7b"
    output_dir = "../models/MLP-LLM-dpo-7b-it"

    best_sweep = {
        "learning_rate": 0.00012149019413774672,
        "batch_size": 32,
        "epochs": 2,
        "weight_decay": 0.1,
        "lr_scheduler_type": "constant",
        "beta": 0.1,
        "optim": "adamw_torch",
    }

    # Attempting to match llama2 SFT settings
    training_args = TrainingArguments(
        report_to="wandb",
        learning_rate=best_sweep["learning_rate"],
        evaluation_strategy="steps",
        eval_steps=50,
        lr_scheduler_type=best_sweep["lr_scheduler_type"],
        optim=best_sweep["optim"],
        output_dir=output_dir,
        per_device_train_batch_size=best_sweep["batch_size"],
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        logging_steps=1,
        num_train_epochs=best_sweep["epochs"],
        max_steps=-1,
        weight_decay=best_sweep["weight_decay"],
        remove_unused_columns=False,
        push_to_hub = True,
        hub_model_id = repo_name,
        hub_token = HF_TOKEN,
    )

    ################
    # Model
    ################
    from unsloth import FastLanguageModel
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name_or_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map="auto",
    )

    model_ref, _ = FastLanguageModel.from_pretrained(
        model_name = model_name_or_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map="auto",
    )

    ################
    # Dataset
    ################

    def parse_data(df):
        full_dict = {
            "prompt": df["prompt"].tolist(),
            "chosen": df["chosen"].tolist(),
            "rejected": df["rejected"].tolist(),
        }

        return full_dict

    df = pd.read_csv(csv_location)
    df = parse_data(df)
    data = Dataset.from_dict(df)

    raw_datasets = HF_Dataset(data)
    train_dataset, eval_dataset = raw_datasets.train_test_split()

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        beta=best_sweep["beta"],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_prompt_length=max_seq_length,
        max_length=max_seq_length,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
