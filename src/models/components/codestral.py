from pathlib import Path

import transformers
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


class Codestral(nn.Module):
    def __init__(
        self,
        model_path: Path,
        quantizer_config: BitsAndBytesConfig = None,
    ):
        super().__init__()
        self.tokenizer: transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast = (
            AutoTokenizer.from_pretrained(model_path)
        )
        self.model: transformers.models.mistral.modeling_mistral.MistralForCausalLM = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                config=quantizer_config,
                device_map="auto",
            )
        )
        print(f"{type(self.model) = }")
        print(f"{type(self.tokenizer) = }")
