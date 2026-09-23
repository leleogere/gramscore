from datetime import datetime
from pathlib import Path
import copy

import datasets
import trl
from loguru import logger
from trl import SFTConfig, SFTTrainer
import transformers
import torch
import numpy as np

MAX_SEQ_LENGTH = 32768  # Supports automatic RoPE Scaling, so choose any number
TEXT_DATASET = (
    "/home/gerel/Documents/datasets/nasap-dataset_chunks=32_overlap=True_min-ratio=0.6_TEXT"
).removesuffix("/")

# %% Load model
tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

# %% Setup tokenizer
chat_template = """
{%- if messages|length == 1 and messages[0]['type'] == 'input' %}
    {{- '[INST]' + messages[0]['content'] + '[/INST]' }}
{%- elif messages|length == 2 and messages[0]['type'] == 'input' and messages[1]['type'] == 'output' %}
    {{- '[INST]' + messages[0]['content'] + '[/INST]' }}
    {%- generation %}
        {{- messages[1]['content'] + eos_token }}
    {%- endgeneration %}
{%- else %}
    {{- raise_exception('Template only supports one input type message, eventually followed by one output type message.') }}
{%- endif %}
""".strip()
print("Using '[control_8]' mistral token as pad token")
tokenizer.pad_token = '[control_8]'
tokenizer.chat_template = chat_template


# %% Prepare the dataset
dataset = datasets.DatasetDict()
for split in ["validation"]:  # ["train", "validation", "test"]:
    print(f"Loading {split} split...")
    dataset[split] = datasets.load_dataset(
        "text",
        data_dir=f"{TEXT_DATASET}/{split}",
        sample_by="document",
    )[
        "train"
    ]  # Needed to flatten the dataset
dataset["validation"] = (
    dataset["validation"].shuffle(seed=10).take(200)
)  # Reduce validation set for faster training


def preprocess_function(samples):
    sep = "\n\n" + "#" * 60 + "\n" + "#" * 60 + "\n\n"
    tokenized = tokenizer.apply_chat_template(
        [
            [
                {"type": "input", "content": (splitted := text.split(sep))[0]},
                {"type": "output", "content": splitted[1]},
            ]
            for text in samples["text"]
        ],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        padding=False,  # Padding is done by the data collator
        truncation=False,
    )
    # Drop items too long
    indices_to_drop = np.where(np.array([len(x) for x in tokenized["input_ids"]]) > MAX_SEQ_LENGTH)[
        0
    ]
    if len(indices_to_drop) > 0:
        logger.warning(
            f"{len(indices_to_drop)} samples of the batch are longer than {MAX_SEQ_LENGTH}, "
            f"dropping them. If this happens too often, consider reducing sequence lengths or "
            f"using a model with a larger context."
        )
        for idx in reversed(indices_to_drop):  # reverse to avoid index shifting
            for key in tokenized:
                tokenized[key].pop(idx)
    # Add labels and mask user tokens
    tokenized["labels"] = [
        [token if is_assistant else -100 for token, is_assistant in zip(input_ids, assistant_masks)]
        for input_ids, assistant_masks in zip(tokenized["input_ids"], tokenized["assistant_masks"])
    ]
    breakpoint()
    return tokenized


breakpoint()
preprocess_function(dataset["validation"][:1])
breakpoint()

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    # num_proc=20,
)

# print(tokenized_dataset["validation"][0])

exit()


# %% Setup an callback to inspect why loss is NaN
class InspectionCallback(transformers.TrainerCallback):
    # def on_pre_optimizer_step(
    #     self,
    #     args: transformers.TrainingArguments,
    #     state: transformers.TrainerState,
    #     control: transformers.TrainerControl,
    #     **kwargs,
    # ):
    #     print("Pre optimizer step")
    #     model = kwargs['model']
    #     # Get min, mean and max of the gradients
    #     gradients = [param.grad for param in model.parameters() if param.grad is not None]
    #     if gradients:
    #         gradients = torch.cat([grad.flatten() for grad in gradients])
    #         print(f"{gradients.min() = }, {gradients.mean() = }, {gradients.max() = }")
    #     else:
    #         print("No gradients to inspect")
    #     # Get min, mean and max of the weights
    #     weights = [param for param in model.parameters()]
    #     if weights:
    #         weights = torch.cat([weight.flatten() for weight in weights])
    #         print(f"{weights.min() = }, {weights.mean() = }, {weights.max() = }")
    #     else:
    #         print("No weights to inspect")

    def on_optimizer_step(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        print("Optimizer step")
        model = kwargs['model']
        # Get min, mean and max of the gradients
        gradients = [param.grad for param in model.parameters() if param.grad is not None]
        if gradients:
            gradients = torch.cat([grad.flatten() for grad in gradients])
            print(f"{gradients.min() = }, {gradients.mean() = }, {gradients.max() = }")
        else:
            print("No gradients to inspect")
        # Get min, mean and max of the weights
        weights = [param for param in model.parameters()]
        if weights:
            weights = torch.cat([weight.flatten() for weight in weights])
            print(f"{weights.min() = }, {weights.mean() = }, {weights.max() = }")
        else:
            print("No weights to inspect")


# %% Prepare the training configuration
output_dir = Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M')}")
output_dir.mkdir(parents=True, exist_ok=False)
training_args = SFTConfig(
    max_seq_length=MAX_SEQ_LENGTH,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    # gradient_checkpointing=True,
    # optim="adam",
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=50,
    report_to="wandb",
    output_dir=str(output_dir),
    logging_dir=str(output_dir / "logs"),
    bf16=True,
)
trainer = SFTTrainer(
    model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    # data_collator=data_collator,
    args=training_args,
    callbacks=[InspectionCallback(), transformers.trainer_callback.PrinterCallback()],
)

# %% Train the model
trainer.train()
