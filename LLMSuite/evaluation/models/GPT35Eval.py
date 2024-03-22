import numpy as np
from openai import OpenAI
from .BaseModelEval import BaseModelEval
import os

class GPT35Eval(BaseModelEval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def generate(self, prompt, system_prompt, template, eval_raw_data,eval_previous_labels):
        try:
            # full_prompt = template.replace("{{ .Prompt }}", prompt).replace("{{ .System }}", system_prompt)
            # print(prompt)
            # print(system_prompt)
            # print(template)
            # response = self.client.chat.completions.create(
            #     engine="gpt-3.5-turbo",  # or "gpt-3.5-turbo" depending on your access
            #     prompt=prompt,
            #     max_tokens=self.config.get("num_predict", 256),
            #     n=self.N_responses,
            #     temperature=self.config.get("temperature", 0.7),
            #     top_p=self.config.get("top_p", 0.9)
            # )
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": system_prompt},
                ]
            )
            # print(response)

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response from GPT-3.5: {e}")
            return None
