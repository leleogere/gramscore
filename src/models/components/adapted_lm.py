from torch import nn
import peft
from transformers import PreTrainedModel


class AdaptedLM(nn.Module):
    def __init__(self, base_lm: PreTrainedModel, peft_config: peft.PeftConfig):
        super().__init__()
        self.adapted_model = peft.get_peft_model(base_lm, peft_config)

    def forward(self, x):
        return self.adapted_model(x)

    # def __getattr__(self, item):
    #     if hasattr(self, item):
    #         return getattr(self, item)
    #     if hasattr(self.model, item):
    #         return getattr(self.model, item)
    #     raise AttributeError(f"{self.__class__.__name__} object has no attribute {item}")
