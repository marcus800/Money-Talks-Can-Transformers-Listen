import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from .BaseModelEval import BaseModelEval
from LLMSuite.evaluation.LSTM import LabelLSTM

class LSTMModelEval(BaseModelEval):
    def __init__(self, config, model_name, N_responses, data_type):
        super().__init__(config, model_name, N_responses)
        self.model = LabelLSTM(input_size=1, hidden_size=128,
                               num_classes=5)  # Ensure these match your model's architecture
        self.data_type = data_type
        state_dict = torch.load(f"../saved_models/{data_type}/lstm_model.pth")
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set the model to evaluation mode
        self.label_map = {0: "MAJOR_INCREASE", 1: "MINOR_INCREASE", 2: "NO_CHANGE", 3: "MINOR_DECREASE",
                          4: "MAJOR_DECREASE"}

    def generate(self, prompt, system_prompt, template, eval_raw_data, eval_previous_labels):
        while len(eval_previous_labels) < 3:
            eval_previous_labels.insert(0, "NO_CHANGE")

        # Adjust label mapping based on your new class mapping
        price_change_map = {v: k for k, v in self.label_map.items()}
        last_3_labels = torch.tensor([[price_change_map[label] for label in eval_previous_labels[-3:]]],
                                     dtype=torch.float).unsqueeze(-1)

        with torch.no_grad():  # Ensure we do not compute gradients
            prediction = self.model(last_3_labels).argmax(dim=1).item()

        predicted_label = self.label_map[prediction]
        return predicted_label
