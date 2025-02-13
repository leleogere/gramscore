from datetime import datetime
from pathlib import Path
import os

import peft
import torch
import trl
import datasets
from loguru import logger
from trl import SFTConfig, SFTTrainer
import transformers

from data.components.patched_tokenizer import PatchedTokenizer

os.environ["XFORMERS_MORE_DETAILS"] = "1"

transformers.set_seed(10)

TEXT_DATASET = (
    "/home/gerel/Documents/datasets/nasap-dataset_chunks=32_overlap=True_min-ratio=0.6_TEXT"
).removesuffix("/")


# %%
# Load dataset
dataset = datasets.DatasetDict()
for split in ["validation"]:  # ["train", "validation", "test"]:
    print(f"Loading {split} split...")
    dataset[split] = datasets.load_dataset(
        "text",
        data_dir=f"{TEXT_DATASET}/{split}",
        sample_by="document",
        streaming=True,
    )[
        "train"
    ]  # Needed to flatten the dataset


# %%
# Define custom chat template
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
# chat_template = """
# {%- if messages|length == 1 and messages[0]['type'] == 'input' %}
#     {{- '<|user|>' + messages[0]['content'] }}
# {%- elif messages|length == 2 and messages[0]['type'] == 'input' and messages[1]['type'] == 'output' %}
#     {{- '<|user|>' + messages[0]['content'] + '<|assistant|>' }}
#     {%- generation %}
#         {{- messages[1]['content'] + eos_token }}
#     {%- endgeneration %}
# {%- else %}
#     {{- raise_exception('Template only supports one input type message, eventually followed by one output type message.') }}
# {%- endif %}
# """.strip()


# %%
# Load tokenizer
# model_id = "facebook/opt-350m"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
model_id = "mistralai/Mistral-7B-v0.3"
# model_id = "bigcode/starcoderbase-1b"
# model_id = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# tokenizer = PatchedTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
max_context_length = transformers.AutoConfig.from_pretrained(model_id).max_position_embeddings
print(f"Max context length: {max_context_length}")

# if tokenizer.name_or_path == "bigcode/starcoderbase-1b":
#     print("Add special tokens to the StarCoder tokenizer")
#     tokenizer.add_special_tokens(
#         {"additional_special_tokens": ["<|input|>", "<|output|>", "<|end|>", "<pad>"]},
#     )
#     tokenizer.pad_token = "<pad>"
#     tokenizer.eos_token = "<|end|>"
#     tokenizer.chat_template = chat_template

# if tokenizer.name_or_path == "microsoft/Phi-3-mini-128k-instruct":
#     tokenizer.pad_token = "<unk>"
#     tokenizer.chat_template = chat_template

if tokenizer.name_or_path == 'mistralai/Mistral-7B-v0.3':
    print("Using '[control_8]' mistral token as pad token")
    tokenizer.pad_token = '[control_8]'
    tokenizer.chat_template = chat_template

# if tokenizer.chat_template is None:
#     print("No chat template, use the Mistral-7B-Instruct-v0.3 one (patched with generation tags)")
#     # tokenizer.chat_template = PatchedTokenizer.from_pretrained(
#     #     "mistralai/Mistral-7B-Instruct-v0.3"
#     # ).chat_template
#     tokenizer.chat_template = chat_template


# %%
# Prepare dataset
SEPARATOR = "\n\n" + "#" * 60 + "\n" + "#" * 60 + "\n\n"


def print_pretty_templated_with_mask(
    tokenizer: transformers.AutoTokenizer,
    sample_conversation: list[dict[str, str]],
):
    from utils.ansi_colors import Colors

    tokenized = tokenizer.apply_chat_template(
        sample_conversation,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=True,
    )
    tokens_ids = tokenized["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(tokens_ids)
    assistant_mask = tokenized["assistant_masks"]
    tokens_string = "".join(tokens)
    mask_string = ""
    for token, mask in zip(tokens, assistant_mask):
        mask_string += (
            (Colors.RED if mask else Colors.BLUE)
            + "⬤"
            # + "━" * (len(token) - 1)
            + "━" * (len(token) - 2)
            + ("╸" if len(token) > 1 else "")
            + Colors.END
        )
    # print(
    # tokenizer.decode(tokens_ids).replace("\n", "¶")
    # )  # better than built string as space are correctly displayed
    print(tokens_string)
    print(mask_string)


# Test the tokenizer
sample = dataset["validation"][0]
input_text, output_text = sample["text"].split(SEPARATOR)
print_pretty_templated_with_mask(
    tokenizer,
    [
        {"type": "input", "content": input_text},
        {"type": "output", "content": output_text},
    ],
)

exit()


def preprocess_function(samples):
    tokenized = tokenizer.apply_chat_template(
        [
            [
                {"type": "input", "content": (splitted := text.split(SEPARATOR))[0]},
                {"type": "output", "content": splitted[1]},
            ]
            for text in samples["text"]
        ],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        padding=False,  # Padding is done by the data collator
        truncation=True,
        max_length=max_context_length,
    )
    tokenized["assistant_masks"] = [list(map(bool, mask)) for mask in tokenized["assistant_masks"]]
    indices_to_drop = [i for i, mask in enumerate(tokenized["assistant_masks"]) if not any(mask)]
    if indices_to_drop:
        logger.warning(
            f"{len(indices_to_drop)} samples of the batch have no output tokens at all withing "
            f"the model's context length, dropping them. If this happens too often, consider "
            f"reducing sequence lengths or using a model with a larger context."
        )
        for idx in reversed(indices_to_drop):  # reverse to avoid index shifting
            for key in tokenized:
                tokenized[key].pop(idx)
    return tokenized


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["validation"].column_names,
)


if True:
    lengths = [len(x) for x in tokenized_dataset["validation"]["input_ids"]]
    print("Min: ", min(lengths))
    print("Max: ", max(lengths))
    print("Mean:", sum(lengths) / len(lengths))


# %%
# Prepare model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_storage="bfloat16",
    ),
)
# Fix the token embeddings size as we added tokens
# model.resize_token_embeddings(len(tokenizer))
# Create the PEFT model
model = peft.get_peft_model(
    model,
    peft_config=peft.LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        inference_mode=False,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(24, 32)),
        layers_pattern="layers",
        # target_modules="all-linear",
        # init_lora_weights="pissa",
    ),
)
model = peft.prepare_model_for_kbit_training(model)

# %%
# Prepare the training configuration
output_dir = Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M')}")
output_dir.mkdir(parents=True, exist_ok=False)
training_args = SFTConfig(
    max_seq_length=max_context_length,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    gradient_checkpointing=True,
    optim="adafactor",
    evaluation_strategy="steps",
    eval_steps=5,
    save_steps=10,
    report_to="wandb",
    output_dir=output_dir,
    logging_dir=output_dir / "logs",
)
trainer = SFTTrainer(
    model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    args=training_args,
)

# %%
# Train the model
trainer.train()

trainer.save_model(output_dir / "final_model")
