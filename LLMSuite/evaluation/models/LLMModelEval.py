from ..llm import LLM
from .BaseModelEval import BaseModelEval

class LLMModelEval(BaseModelEval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = LLM(self.name, self.config)

    def generate(self, prompt, system_prompt, template, eval_raw_data,eval_previous_labels):
        return self.llm.generate(prompt=prompt, system_prompt=system_prompt, template=template)
