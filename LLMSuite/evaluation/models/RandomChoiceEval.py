import numpy as np
from .BaseModelEval import BaseModelEval
import joblib


class RandomChoiceEval(BaseModelEval):
    def __init__(self, config, model_name, N_responses,data_type, weighted=False):
        super().__init__(config, model_name, N_responses)
        self.weighted = weighted
        if self.weighted:
            label_distribution = joblib.load(f"../saved_models/{data_type}/label_distribution.pkl")
            self.weights = [label_distribution.get(label, 0) for label in
                            ["MAJOR_DECREASE", "MINOR_DECREASE", "NO_CHANGE", "MINOR_INCREASE", "MAJOR_INCREASE"]]

    def generate(self, prompt, system_prompt, template, eval_raw_data, eval_previous_labels):
        predictions = ["MAJOR_DECREASE", "MINOR_DECREASE", "NO_CHANGE", "MINOR_INCREASE", "MAJOR_INCREASE"]
        if self.weighted:
            return np.random.choice(predictions, p=list(self.weights))
        else:
            return np.random.choice(predictions)
