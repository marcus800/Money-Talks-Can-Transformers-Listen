import os, yaml, wandb
import pandas as pd

from tqdm import tqdm
import numpy as np

from llm import LLM

TEMPLATE = """
[INST]
<<SYS>>{{ .System }}<</SYS>>
Make a prediction based on the following excerpt:
{{ .Prompt }}
[/INST]
Based on the given excerpt, the change in stock price could be categorized as: 
"""

SYSTEM_PROMPT = """You are an expert at predicting whether a stock price that is correlated with the a piece of text, will increase, decrease, or stay the same, given an excerpt spoken by an important figure in finance. The change in stock price can be one of the following: MAJOR_DECREASE, MINOR_DECREASE, NO_CHANGE, MINOR_INCREASE, MAJOR_INCREASE. Make sure your prediction is based on the meaning of the excerpt and not on your prior knowledge."""

def parse_response(response):

    response = response.lower()

    if any(label in response for label in ["major_decrease", "major decrease"]):
        return "MAJOR_DECREASE"

    if any(label in response for label in ["minor_decrease", "minor decrease"]):
        return "MINOR_DECREASE"
    
    if any(label in response for label in ["no_change", "no change"]):
        return "NO_CHANGE"
    
    if any(label in response for label in ["minor_increase", "minor increase"]):
        return "MINOR_INCREASE"
    
    if any(label in response for label in ["major_increase", "major increase"]):
        return "MAJOR_INCREASE"
    
    return "UNKNOWN"

def evaluate(config=None):
    """
    Evaluate a single model on a dataset
    Returns a dictionary with the following keys
    - accuracy
    - f1
    - precision
    - recall
    - percentage of each class
    """

    with wandb.init(config=config):
        # set sweep configuration
        config = wandb.config

        data = pd.read_csv("eval_data/evaluation_data.csv")
        model_name = "mlpllm"
        N_responses = 5

        llm_config = {
            "temperature" : config.temperature,
            "top_k" : config.top_k,
            "top_p" : config.top_p,
            "num_predict" : config.num_predict,
        }

        model = LLM(model_name, llm_config)

        results = []

        for i in tqdm(range(len(data)), desc=f"Evaluating {model_name}"):
            sentence = data.iloc[i]["sentence"]
            label = data.iloc[i]["label"]
            
            responses = [ model.generate(prompt=sentence, system_prompt=SYSTEM_PROMPT, template=TEMPLATE) for _ in range(N_responses) ]

            predicted_labels = [ parse_response(response) for response in responses ]
            predicted_label = max(set(predicted_labels), key=predicted_labels.count)

            results += [(sentence, label, responses, predicted_label)]

        results = pd.DataFrame(results, columns=["sentence", "label", "response", "predicted_label"])
        results.to_csv(f"outputs/{model_name}_output.csv")

        # Total accuracy of correctly labeled sentences
        accuracy_total = np.mean(results["label"] == results["predicted_label"])

        # Accuracy for each class
        accuracy_per_class = {
            label: np.mean(results[results["label"] == label]["label"] == results[results["label"] == label]["predicted_label"])
            for label in ["MAJOR_DECREASE", "MINOR_DECREASE", "NO_CHANGE", "MINOR_INCREASE", "MAJOR_INCREASE"]
        }

        # Precision for each class 
        precision = {
            label: np.mean(
                (results["label"] == label) & (results["predicted_label"] == label)
            ) / np.mean((results["predicted_label"] == label))
            for label in ["MAJOR_DECREASE", "MINOR_DECREASE", "NO_CHANGE", "MINOR_INCREASE", "MAJOR_INCREASE"]
        }

        # Recall for each class 
        recall = {
            label: np.mean(
                (results["label"] == label) & (results["predicted_label"] == label)
            ) / np.mean(results["label"] == label)
            for label in ["MAJOR_DECREASE", "MINOR_DECREASE", "NO_CHANGE", "MINOR_INCREASE", "MAJOR_INCREASE"]
        }

        # F1 for each class 
        f1 = {
            label: 2 * (precision[label] * recall[label]) / (precision[label] + recall[label])
            for label in ["MAJOR_DECREASE", "MINOR_DECREASE", "NO_CHANGE", "MINOR_INCREASE", "MAJOR_INCREASE"]
        }

        # Replace NaNs with 0
        precision = {label: 0 if np.isnan(precision[label]) else precision[label] for label in precision}
        recall = {label: 0 if np.isnan(recall[label]) else recall[label] for label in recall}
        f1 = {label: 0 if np.isnan(f1[label]) else f1[label] for label in f1}
        
        metrics = {
            "accuracy_total": accuracy_total,
            "accuracy_per_class": accuracy_per_class,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "percentage_of_each_class": results["label"].value_counts(normalize=True).to_dict()
        }

        wandb.log(metrics)

        return metrics

if __name__ == "__main__":

    SWEEP_COUNT = 1000000

    os.environ["WANDB_PROJECT"] = "MLP-LLM-EVALUATION-SWEEP"
    os.environ["WANDB_LOG_MODEL"] = "MLP-LLM-13b"

    sweep_config = yaml.load(open("sweep_config.yaml"), Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(sweep_config, project=os.environ.get("WANDB_PROJECT"))

    wandb.agent(sweep_id, evaluate, count=SWEEP_COUNT)

