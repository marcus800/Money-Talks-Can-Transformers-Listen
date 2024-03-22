import requests, json
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append("../../evaluation")
from llm import LLM

from jinja2 import Environment, FileSystemLoader

TEMPLATE_NAME = "S&P500_gpt.j"

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

def get_difference(label, predicted_label):

    if label == predicted_label:
        return 0.0
    
    if label == "MAJOR_DECREASE":
        if predicted_label == "MINOR_DECREASE":
            return 0.5
        if predicted_label == "NO_CHANGE":
            return 1.0
        if predicted_label == "MINOR_INCREASE":
            return 1.5
        if predicted_label == "MAJOR_INCREASE":
            return 2.0
        
    if label == "MINOR_DECREASE":
        if predicted_label == "MAJOR_DECREASE":
            return 0.5
        if predicted_label == "NO_CHANGE":
            return 0.5
        if predicted_label == "MINOR_INCREASE":
            return 1.0
        if predicted_label == "MAJOR_INCREASE":
            return 1.5
        
    if label == "NO_CHANGE":
        if predicted_label == "MAJOR_DECREASE":
            return 1.0
        if predicted_label == "MINOR_DECREASE":
            return 0.5
        if predicted_label == "MINOR_INCREASE":
            return 0.5
        if predicted_label == "MAJOR_INCREASE":
            return 1.0
        
    if label == "MINOR_INCREASE":
        if predicted_label == "MAJOR_DECREASE":
            return 1.5
        if predicted_label == "MINOR_DECREASE":
            return 1.0
        if predicted_label == "NO_CHANGE":
            return 0.5
        if predicted_label == "MAJOR_INCREASE":
            return 0.5
        
    if label == "MAJOR_INCREASE":
        if predicted_label == "MAJOR_DECREASE":
            return 2.0
        if predicted_label == "MINOR_DECREASE":
            return 1.5
        if predicted_label == "NO_CHANGE":
            return 1.0
        if predicted_label == "MINOR_INCREASE":
            return 0.5

def make_response(response, label):
    predicted_label = parse_response(response)
    return {
        "response": response,
        "label": label,
        "predicted_label": predicted_label,
        "difference": get_difference(label, predicted_label),
    }

if __name__ == "__main__":

    ### Load model to LLM object 
    model_repo = "seanmemery/MLP-FinLLM-7b-it"
    config = {
        "ctx_n": 512,
        "temperature": 1.1,
        "top_k": 40,
        "top_p": 1.0,
        "num_predict": 512,
        "model_source": "hf",
    }
    llm = LLM(model_repo, config)

    template = Environment(loader=FileSystemLoader('../../templates')).get_template(TEMPLATE_NAME)

    ### Load "data.csv" into a pandas dataframe
    df = pd.read_csv("data.csv")

    ### Have the LLM make reasoning outputs and a label prediction for each data point
    new_df = []
    N_sentences = 3
    for i in tqdm(range(len(df)), desc="Generating Instruction Data"):
        label = df.at[i, "label"]
        
        excerpt = df.at[i, "sentence"] 
        for j in range(1, N_sentences):
            if i - j >= 0:
                excerpt = df.at[i - j, "sentence"] + " " + excerpt

        full_prompt = template.render(sentence=excerpt)

        # Add space for reasoning output
        full_prompt += """\nThe reasoning for my prediction is given below\n###Reasoning\nBased on the above excerpt """

        llm_template = """{{ .Prompt }}"""

        responses = []
        attempts = 0
        while len(responses) < 2:
            attempts += 1
            response = llm.generate(prompt=full_prompt, system_prompt="", template=llm_template)
            response = make_response(response, label)

            if response["predicted_label"] == "UNKNOWN":
                continue

            if len(responses) == 0 or response["difference"] != responses[0]["difference"]:
                responses.append(response)

        print(f"Attempted {attempts} times to get two different responses for prompt {i}.")

        ## Choose a response based on difference 
        if responses[0]["difference"] < responses[1]["difference"]:
            chosen = responses[0]
            rejected = responses[1]
        else:
            chosen = responses[1]
            rejected = responses[0]

        new_df.append({
            "prompt": full_prompt,
            "chosen": chosen["response"],
            "rejected": rejected["response"],
        })

        if i % 10 == 0:
            pd.DataFrame(new_df).to_csv("dpo_data.csv", index=False)




