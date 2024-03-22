from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
import pandas as pd
import os


def get_instruction(instruction_template, sentence):
    return instruction_template.render(
        sentence=sentence
    )

def label_row(price_change, mean, std_dev):
    if price_change > mean + std_dev:
        return "MAJOR_INCREASE"
    elif price_change > mean:
        return "MINOR_INCREASE"
    elif price_change < mean - std_dev:
        return "MAJOR_DECREASE"
    elif price_change < mean:
        return "MINOR_DECREASE"
    else:
        return "NO_CHANGE"

'''
Using Labled Data! 
'''
DATA_PATH = "../processed_data/labeled/"
FILE_FOR_EVALUATION = "January 26, 2022.csv"
TARGET_DATAS = ["USA500.IDXUSD", "EURUSD"]
TEMPLATE_NAMES = ["S&P_gpt.j", "FX_gpt.j", "joint_gpt"]
# EXPERIMENT_NAMES = ["S&P", "FX", "joint"]
EXPERIMENT_NAMES = ["S&P", "FX"]

for experiment in EXPERIMENT_NAMES:
    if experiment == "S&P":
        TARGET_DATA = TARGET_DATAS[0]
        TARGET_COLUMN = f"{TARGET_DATA} Close Difference"
        TEMPLATE_NAME = TEMPLATE_NAMES[0]
    elif experiment == "FX":
        TARGET_DATA = TARGET_DATAS[1]
        TARGET_COLUMN = f"{TARGET_DATA} Close Difference"
        TEMPLATE_NAME = TEMPLATE_NAMES[1]
    print(experiment)


    all_days = []
    plot = True
    import matplotlib.pyplot as plt;
    import seaborn as sns
    plt.figure(figsize=(10, 6))  # Set the figure size for the plot
    plt.rcParams.update({'font.size': 12})

    for f in os.listdir(DATA_PATH):
        if f.endswith(".csv") and f != FILE_FOR_EVALUATION:
            training_data_day = pd.read_csv(DATA_PATH + f)
            training_data_day = training_data_day.dropna(subset=[TARGET_COLUMN], axis=0)
            training_data_day = training_data_day.reset_index(drop=True)

            std_dev_price_change = training_data_day[TARGET_COLUMN].std()
            #Hard coding to 0
            mean_price_change = 0
            if plot:
                sns.histplot(training_data_day[TARGET_COLUMN], kde=True, label=f)

            for i, row in training_data_day.iterrows():
                training_data_day.at[i, "label"] = label_row(row[TARGET_COLUMN], mean_price_change, std_dev_price_change)

            all_days.append(training_data_day)
    if plot:
        # plt.legend()  # Show the legend to identify each plot
        plt.savefig(f'plot_{experiment}.pdf')
        plt.show()  # Display the plot

    evaluation_data = pd.read_csv(DATA_PATH + FILE_FOR_EVALUATION)
    evaluation_data = evaluation_data.dropna(subset=[TARGET_COLUMN], axis=0)
    evaluation_data = evaluation_data.reset_index(drop=True)
    std_dev_price_change = evaluation_data[TARGET_COLUMN].std()
    mean_price_change = 0
    for i, row in evaluation_data.iterrows():
        evaluation_data.at[i, "label"] = label_row(row[TARGET_COLUMN], mean_price_change, std_dev_price_change)

    training_data = pd.concat(all_days)
    # Print percentage of each label
    print(training_data["label"].value_counts(normalize=True))

    # Reform data to just have sentence, label, price change
    training_data = training_data[["Sentence", "label", TARGET_COLUMN]]
    training_data = training_data.rename(columns={TARGET_COLUMN: "price_change"})
    training_data = training_data.reset_index(drop=True)
    training_data.to_csv(f"training/train_data/{experiment}/data.csv", index=False)

    evaluation_data = evaluation_data[["Sentence", "label", TARGET_COLUMN]]
    evaluation_data = evaluation_data.rename(columns={TARGET_COLUMN: "price_change"})
    evaluation_data = evaluation_data.reset_index(drop=True)
    evaluation_data.to_csv(f"evaluation/eval_data/{experiment}/evaluation_data.csv", index=False)

    instruction_template = Environment(loader=FileSystemLoader('templates')).get_template(TEMPLATE_NAME)

    ### TRAINING DATA
    HF_dataset = []
    num_sentences = 3
    for i, d in tqdm(training_data.iterrows(), desc="Generating Instruction Data", total=len(training_data)):
        answer = f"Based on the given excerpt, the change could be categorized as: {d['label']}."
        excerpt = training_data.at[i, "Sentence"]
        for j in range(1, num_sentences):
            if i - j >= 0:
                excerpt = training_data.at[i - j, "Sentence"] + " " + excerpt
        text = get_instruction(instruction_template, excerpt) + "\n" + answer
        HF_dataset.append({"text": text})
    HF_dataset = pd.DataFrame(HF_dataset)
    HF_dataset.to_csv(f"training/train_data/{experiment}/instruction_data.csv", index=False)

    ### EVALUATION DATA
    HF_dataset = []
    num_sentences = 3
    for i, d in tqdm(evaluation_data.iterrows(), desc="Generating Instruction Data", total=len(evaluation_data)):
        answer = f"Based on the given excerpt, the change could be categorized as: {d['label']}."
        excerpt = evaluation_data.at[i, "Sentence"]
        for j in range(1, num_sentences):
            if i - j >= 0:
                excerpt = evaluation_data.at[i - j, "Sentence"] + " " + excerpt
        text = get_instruction(instruction_template, excerpt) + "\n" + answer
        HF_dataset.append({"text": text})
    HF_dataset = pd.DataFrame(HF_dataset)
    HF_dataset.to_csv(f"evaluation/eval_data/{experiment}/instruction_data.csv", index=False)

    #RAW EVAL DATA
    evaluation_data_raw = pd.read_csv(DATA_PATH + FILE_FOR_EVALUATION)
    evaluation_data_raw.to_csv(f"evaluation/eval_data/{experiment}/raw_eval.csv", index=False)

    #RAW TRAINING DATA
    training_data = pd.concat(all_days)
    training_data.to_csv(f"training/train_data/{experiment}/raw_training.csv", index=False)


