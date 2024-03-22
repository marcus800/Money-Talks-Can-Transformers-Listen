import os
from tqdm import tqdm
from transformers import TrainingArguments
from datasets import Dataset
import pandas as pd

from trl import SFTTrainer

from dataset import HF_Dataset

tqdm.pandas()

if __name__ == "__main__":

    HF_TOKEN = os.environ.get("HF_WRITE_TOKEN")
    assert HF_TOKEN is not None, "You must set the HF_WRITE_TOKEN environment variable."

    DATA_TARGET = "FX"

    os.environ["WANDB_PROJECT"] = "MLP-LLM"
    os.environ["WANDB_LOG_MODEL"] = "MLP-LLM-7b"

    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    csv_location = f"train_data/{DATA_TARGET}/instruction_data.csv"
    dataset_text_field = "text"
    max_seq_length = 512
    repo_name = f"seanmemery/MLP-FinLLM-7b-it-{DATA_TARGET}"
    output_dir = f"../models/MLP-LLM-7b-it-{DATA_TARGET}"

    best_sweep = {
        "learning_rate": 0.0025177606136092684,
        "batch_size": 32,
        "weight_decay": 0.05,
        "lr_scheduler_type": "polynomial",
    }

    # Attempting to match llama2 SFT settings
    training_args = TrainingArguments(
        report_to="wandb",
        learning_rate=best_sweep["learning_rate"],
        evaluation_strategy="steps",
        eval_steps=50,
        lr_scheduler_type=best_sweep["lr_scheduler_type"],
        optim="adamw_torch",
        save_strategy="no",
        output_dir=output_dir,
        per_device_train_batch_size=best_sweep["batch_size"],
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        logging_steps=1,
        num_train_epochs=4,
        max_steps=-1,
        weight_decay=best_sweep["weight_decay"],
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

    ################
    # Dataset
    ################
    df = pd.read_csv(csv_location) 
    data = Dataset.from_pandas(df)
    raw_datasets = HF_Dataset(data)
    train_dataset, eval_dataset = raw_datasets.train_test_split()

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
    )
    trainer.train()
    trainer.save_model(output_dir)

    #trainer.push_to_hub()
