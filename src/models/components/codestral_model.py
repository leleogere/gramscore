from pathlib import Path

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


class Codestral(nn.Module):
    def __init__(
        self,
        model_path: Path,
        bnp_config: BitsAndBytesConfig = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=bnp_config,
            device_map="auto",
        )
        print(f"{type(self.model) = }")
        print(f"{type(self.tokenizer) = }")
