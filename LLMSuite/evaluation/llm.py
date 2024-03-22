import requests, json, torch, gc
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

import vllm 

class LLM():
    def __init__(self, model_name_or_repo, config):
    
        self.CTX_N = config["ctx_n"]
        self.MODEL_OPTIONS = {
            "temperature": config["temperature"],
            "top_k": config["top_k"],
            "top_p": config["top_p"],
            "num_predict": config["num_predict"],
        }
        self.url = ""

        self.source = config["model_source"]
        self.model_name_or_repo = model_name_or_repo
        if self.source == "ollama":
            self.setup_ollama()
        else:
            self.model = None
            self.setup_hf()

    def __del__(self):

        # Delete the llm object and free the memory
        destroy_model_parallel()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()

    def setup_hf(self):
        self.model = vllm.LLM(model=self.model_name_or_repo, dtype="bfloat16")
                
    def setup_ollama(self):
        url = "http://localhost:11434/api/tags"
        response = requests.get(url)
        models = json.loads(response.text)["models"]
        models = [x["name"] for x in models]
        if self.model_name not in models:
            print(f"Model {self.model_name} not found in Ollama pool, attempting download...")
            url = "http://localhost:11434/api/pull"
            myobj = {
                "name": self.model_name,
            }
            resp = requests.post(url, json = myobj)

            if resp.status_code != 200:
                raise Exception(f"Request failed with status {resp.status_code} and message {resp.text}")
                        
        else:
            print(f"Model {self.model_name} found in Ollama pool, continuing...")

        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt, system_prompt, template):

        message = {
            "model": self.model_name_or_repo,
            "json": True,
            "system": system_prompt,
            "stream": False,
            "prompt": prompt,
            "options": self.MODEL_OPTIONS,
            "num_ctx": self.CTX_N,
            "template": template,
        }

        if self.source == "ollama":
            return self.handle_ollama(message)
        else:
            return self.handle_hf(message)
        
    def handle_ollama(self, message):
        response = requests.post(self.url, json=message)
        return json.loads(response.text)["response"]

    def handle_hf(self, message):

        full_prompt = message["template"].replace("{{ .System }}", message["system"])
        full_prompt = full_prompt.replace("{{ .Prompt }}", message["prompt"])

        sampling_params = vllm.SamplingParams(
            n = 1,
            temperature = message["options"]["temperature"],
            top_k = message["options"]["top_k"],
            top_p = message["options"]["top_p"],
            max_tokens = message["options"]["num_predict"],
        )

        outputs = self.model.generate(full_prompt, sampling_params)

        return outputs[0].outputs[0].text


        # gen_config = GenerationConfig.from_dict(        
        #     {
        #         "temperature": message["options"]["temperature"],
        #         "top_k": message["options"]["top_k"],
        #         "top_p": message["options"]["top_p"],
        #         "max_new_tokens": message["options"]["num_predict"],
        #         "do_sample": True,
        #     }
        # )
        # self.model.generation_config = gen_config
        
        # with torch.no_grad():

        #     response = self.model.generate(
        #         self.tokenizer.encode(full_prompt, return_tensors="pt").cuda()
        #     )
        #     response = self.tokenizer.decode(response[0], skip_special_tokens=True).split("[/INST]\n")[-1]

        # return response


