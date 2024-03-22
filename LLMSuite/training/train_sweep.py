import os, yaml
from tqdm import tqdm
from transformers import TrainingArguments, integrations
from datasets import Dataset
import wandb
import pandas as pd

import numpy as np
import evaluate

from trl import SFTTrainer

from dataset import HF_Dataset

tqdm.pandas()

if __name__ == "__main__":

    SWEEP_COUNT = 1000000

    os.environ["WANDB_PROJECT"] = "MLP-LLM-SWEEP"
    os.environ["WANDB_LOG_MODEL"] = "MLP-LLM-7b"

    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    csv_location = "train_data/instruction_data.csv"
    dataset_text_field = "text"
    max_seq_length = 512
    output_dir = "../models/MLP-LLM7b-sweep-it"

    ################
    # Sweep Config
    ################

    sweep_config = yaml.load(open("sweep_config.yaml"), Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(sweep_config, project=os.environ.get("WANDB_PROJECT"))

    def train(config=None):

        with wandb.init(config=config):
            # set sweep configuration
            config = wandb.config

            training_args = TrainingArguments(
                save_strategy="no",
                report_to="wandb",
                learning_rate=config.learning_rate,
                lr_scheduler_type=config.lr_scheduler_type,
                optim="adamw_torch",
                output_dir=output_dir,
                per_device_train_batch_size=config.batch_size,
                per_device_eval_batch_size=1,
                gradient_checkpointing=True,
                logging_steps=1,
                num_train_epochs=4,
                max_steps=-1,
                weight_decay=config.weight_decay,
                push_to_hub = False,
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

            wandb.log({'eval_loss': trainer.evaluate().get("eval_loss")})

    wandb.agent(sweep_id, train, count=SWEEP_COUNT)
