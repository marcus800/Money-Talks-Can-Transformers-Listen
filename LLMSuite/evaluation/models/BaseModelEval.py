class BaseModelEval:
    def __init__(self, config, model_name, N_responses):
        self.config = config
        self.name = model_name
        self.N_responses = N_responses

    def generate(self, prompt, system_prompt, template, eval_raw_data, eval_previous_labels):
        raise NotImplementedError("This method should be implemented by subclasses.")
