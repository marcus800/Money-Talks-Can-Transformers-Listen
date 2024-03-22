import joblib
from .BaseModelEval import BaseModelEval
import pandas as pd

class LRModelEval(BaseModelEval):
    def __init__(self, config, model_name, N_responses, data_type):
        super().__init__(config, model_name, N_responses)
        self.model = joblib.load(f"../saved_models/{data_type}/softmax_regression_model.pkl")
        self.label_map = {2: "MAJOR_INCREASE", 1: "MINOR_INCREASE", 0: "NO_CHANGE", -1: "MINOR_DECREASE",
                          -2: "MAJOR_DECREASE"}
        self.price_change_map = {
            "MAJOR_INCREASE": 2,
            "MINOR_INCREASE": 1,
            "NO_CHANGE": 0,
            "MINOR_DECREASE": -1,
            "MAJOR_DECREASE": -2
        }

    def generate(self, prompt, system_prompt, template, eval_raw_data, eval_previous_labels):
        while len(eval_previous_labels) < 3:
            eval_previous_labels.insert(0, "NO_CHANGE")

        last_3_labels = [self.price_change_map[label] for label in eval_previous_labels[-3:]]
        features_df = pd.DataFrame([last_3_labels], columns=['shift_1', 'shift_2', 'shift_3'])

        prediction = self.model.predict(features_df)
        predicted_label = self.label_map[prediction[0]]
        return predicted_label
