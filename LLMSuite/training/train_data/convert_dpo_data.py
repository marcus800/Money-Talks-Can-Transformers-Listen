import pandas as pd
from jinja2 import Environment, FileSystemLoader

dpo_data = pd.read_csv("dpo_data.csv")
new_template_data = pd.read_csv("S&P/instruction_data.csv")

for i in range(len(dpo_data)):
    new_template = new_template_data.iloc[i]["text"]

    # Get text between [INST] and [\INST]
    inst_start = new_template.find("[INST]")
    inst_end = new_template.find("[/INST]")
    inst_text = new_template[inst_start:inst_end] + "[/INST]" + "\n### Reasoning\nThe reasoning behind my prediction is "

    dpo_data.iloc[i]["prompt"] = inst_text

dpo_data.to_csv("dpo_data.csv", index=False)