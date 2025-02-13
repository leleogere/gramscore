from pathlib import Path

import transformers


model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=Path(
        "/home/gerel/Documents/transcription/gramscore/models/first_working_training_checkpoint-1200"
    ),
)
