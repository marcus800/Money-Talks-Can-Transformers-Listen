import yaml, argparse, json
import pandas as pd
from tqdm import tqdm
import os

from jinja2 import Environment, FileSystemLoader
from helper_funs import parse_response, eval_metrics

# creating code to evaluate everything on any prompt and any selection of models and any test data
out_file = 'both_randoms'
USE_LLMS = False
USE_GPT = False


def evaluate(models, eval_processed_data, eval_raw_data, instruction_data, template, run_name):
    results = {}
    metrics = {}
    for model in models:
        config = model.config
        N_responses = model.N_responses
        model_name = model.name

        # print(instruction_data.iloc[0]["text"])
        # print(instruction_data.iloc[10]["text"])
        print(f"Model {model_name}")
        # sentence = eval_processed_data.iloc[10]["Sentence"]
        # print(f"{template.render(sentence=sentence)}")
        results[model_name] = []

        for i in tqdm(range(len(eval_processed_data)), desc=f"Evaluating {model_name}"):
            sentence = eval_processed_data.iloc[i]["Sentence"]
            label = eval_processed_data.iloc[i]["label"]
            previous_labels = eval_processed_data["label"].iloc[:i].tolist()
            # insturtion = instruction_data.iloc[i]["text"]

            prompt = template.render(sentence=sentence)
            llm_template = """{{ .Prompt }}"""
            responses = [model.generate(prompt=sentence, system_prompt=prompt, template=llm_template,
                                        eval_raw_data=eval_raw_data, eval_previous_labels=previous_labels) for _ in
                         range(N_responses)]

            predicted_labels = [parse_response(response) for response in responses]
            predicted_label = max(set(predicted_labels), key=predicted_labels.count)

            results[model_name] += [(sentence, label, responses, predicted_label)]

        metrics[model_name] = eval_metrics(results[model_name])
    print(metrics)

    base_path = 'results'
    existing_runs = [int(f.split(run_name)[-1].split('_')[0]) for f in os.listdir(base_path) if
                     f.startswith(run_name) and ('_results.json' in f or '_metrics.json' in f)]
    next_run_number = max(existing_runs) + 1 if existing_runs else 0

    with open(f"{base_path}/{run_name}{next_run_number}_results.json", "w") as f:
        json.dump(results, f, indent=4)
    with open(f"{base_path}/{run_name}{next_run_number}_metrics.json", "w") as f:
        print(f"Saved to :{base_path}/{run_name}{next_run_number}_metrics.json")
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    with open("eval_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    llm_models_config = config['llm']
    llm_N_choices = config['N_responses_llm']
    template_name = config['template']
    #TODO take care that the tabmplate_name is the same for both!
    data_type = "FX"
    template_name = f"{data_type}_gpt.j"

    template = Environment(loader=FileSystemLoader('../templates')).get_template(template_name)

    eval_processed_data = pd.read_csv(f"eval_data/{data_type}/evaluation_data.csv")
    # instruction_data = pd.read_csv(f"eval_data/{data_type}/instruction_data.csv")
    instruction_data = None
    eval_raw_data = pd.read_csv(f"eval_data/{data_type}/raw_eval.csv")

    llm_models = []

    if USE_LLMS:
        from models.LLMModelEval import LLMModelEval

        for model_name in config['llm_models']:
            model_config = llm_models_config.copy()  # Copy the general LLM config
            llm_models.append(LLMModelEval(model_config, model_name, llm_N_choices))

    from models.RandomChoiceEval import RandomChoiceEval
    from models.LSTMModelEval import LSTMModelEval
    from models.LRModelEval import LRModelEval

    random_model = RandomChoiceEval(None, "RandomChoice Weighed", 1, data_type, weighted=True)
    random_model2 = RandomChoiceEval(None, "RandomChoice Not Weighted", 1, data_type, weighted=False)
    rl_model = LRModelEval(None, "LR", 1, data_type)
    lstm_model = LSTMModelEval(None, "LSTM1", 1, data_type)

    if USE_GPT:
        from models.GPT35Eval import GPT35Eval
        model_config = {}
        model_name = "gpt3.5"
        llm_models.append(GPT35Eval(model_config, model_name, 3))

    models = llm_models + [random_model2] + [random_model] + [rl_model] + [lstm_model]

    run_name = f"{out_file}-{data_type}"
    evaluate(models, eval_processed_data, eval_raw_data, instruction_data,template, run_name)
