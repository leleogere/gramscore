from pathlib import Path
from typing import Any
from typing import Mapping

import peft
import torch
import trl
import datasets
from loguru import logger
from trl import SFTConfig, SFTTrainer
from pprint import pprint
import transformers

from data.components.patched_tokenizer import PatchedTokenizer
from utils.ansi_colors import Colors

transformers.set_seed(10)

TEXT_DATASET = (
    "/home/gerel/Documents/datasets/nasap-dataset_chunks=32_overlap=True_min-ratio=0.6_TEXT"
).removesuffix("/")

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

# %%
# Load dataset
dataset = datasets.DatasetDict()
for split in ["train"]:  # , "validation", "test"]:
    dataset[split] = datasets.load_dataset(
        "text",
        data_dir=f"{TEXT_DATASET}/{split}",
        split=split,
        sample_by="document",
    )

SEPARATOR = "\n\n" + "#" * 60 + "\n" + "#" * 60 + "\n\n"


def split_text(rows: dict[str, list[str]], sep: str = SEPARATOR) -> dict[str, list[str]]:
    """
    Split x[text] by sep into performance and score.
    """

    def split_single_text(text: str) -> tuple[str, str]:
        parts = text.split(sep)
        assert len(parts) == 2, f"Expected 2 parts, got {len(parts)}"
        return parts

    split_texts = list(map(split_single_text, rows["text"]))
    return {
        "performance": [s[0] for s in split_texts],
        "score": [s[1] for s in split_texts],
    }


dataset = dataset.map(split_text, batched=True, remove_columns=["text"])

# %%
# Explore dataset
# pprint(dataset)
# print(len(dataset["train"]))
# for key, value in dataset["train"][0].items():
#     print(f"{key} {str(type(value))}: {value}")


# %%
# Load tokenizer
model_id = "facebook/opt-350m"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"
# model_id = "mistralai/Mistral-7B-v0.3"
# tokenizer = PatchedTokenizer.from_pretrained(model_id)
tokenizer = PatchedTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
max_context_length = transformers.AutoConfig.from_pretrained(model_id).max_position_embeddings
print(f"Max context length: {max_context_length}")

if tokenizer.pad_token is None:
    print("No pad token, use '[control_8]' mistral token as pad token")
    assert tokenizer.name_or_path == 'mistralai/Mistral-7B-v0.3', "Only Mistral tokenizer supported"
    tokenizer.pad_token = '[control_8]'

if tokenizer.chat_template is None:
    print("No chat template, use the Mistral-7B-Instruct-v0.3 one (patched with generation tags)")
    # tokenizer.chat_template = PatchedTokenizer.from_pretrained(
    #     "mistralai/Mistral-7B-Instruct-v0.3"
    # ).chat_template
    tokenizer.chat_template = chat_template


def print_pretty_templated_with_mask(
    tokenizer: transformers.AutoTokenizer,
    sample_conversation: list[dict[str, str]],
):
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


if False:
    sample_conversation = [
        [
            # fmt: off
        {"type": "input", "content": dataset["train"]["performance"][i]},
        {"type": "output", "content": dataset["train"]["score"][i]},
            # fmt: on
        ]
        for i in range(min(3000, len(dataset["train"])))
    ]
    tokenized = tokenizer.apply_chat_template(
        sample_conversation,
        tokenize=True,
        return_dict=True,
    )
    lengths = [len(x) for x in tokenized["input_ids"]]
    print("Min: ", min(lengths))
    print("Max: ", max(lengths))
    print("Mean:", sum(lengths) / len(lengths))
    # For 3000 samples:
    # Min:  6591
    # Max:  22880
    # Mean: 14614.993718592965


if False:
    sample_conversation = [
        # fmt: off
        {"type": "input", "content": "Note(p=14, t=0, d=234, v=12)\nNote(p=15, t=684, d=256, v=15)\nNote(p=14, t=750, d=528, v=45)"},
        {"type": "output", "content": "X:1\nT:Untitled\nM:4/4\nK:C\nC2 E2 G2 | C2 E2 G2 | C2 E2 G2 | C2 E2 G2 |"},
        # fmt: on
    ]
    print_pretty_templated_with_mask(tokenizer, sample_conversation)

if False:
    sample_conversation = [
        # fmt: off
        {"type": "input", "content": "What does Big Data really do with our data?"},
        {"type": "output", "content": """"Big Data" isn't an entity so it does nothing with our data. "Big Data" refers to the concept of using large amounts of data to draw conclusions or direct action. Some company might collect "Big Data" about their customers and then try to draw conclusions about their customer base, such as what products to carry or advertise. Similarly selling that data to other companies would be profitable, and buying such data could in turn help grow a business."""},
        # fmt: on
    ]
    print_pretty_templated_with_mask(tokenizer, sample_conversation)


# %%
# Prepare dataset
def preprocess_function(samples):
    # print(samples["performance"][0])
    # print(samples["score"][0])
    tokenized = tokenizer.apply_chat_template(
        [
            [
                {"type": "input", "content": performance},
                {"type": "output", "content": score},
            ]
            for performance, score in zip(samples["performance"], samples["score"])
        ],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        padding=False,  # Padding is done by the data collator
        truncation=True,
        max_length=max_context_length,
    )
    tokenized["text_input"] = samples["performance"]
    tokenized["text_output"] = samples["score"]
    # tokenized["assistant_masks"] = torch.LongTensor(tokenized["assistant_masks"])
    # tokenized["output_starts"] = tokenized["assistant_masks"].argmax(dim=1)
    tokenized["assistant_masks"] = [list(map(bool, mask)) for mask in tokenized["assistant_masks"]]
    indices_to_drop = [i for i, mask in enumerate(tokenized["assistant_masks"]) if not any(mask)]
    if indices_to_drop:
        logger.warning(
            f"{len(indices_to_drop)} samples of the batch have no output tokens at all withing "
            f"the model's context length, dropping them. If this happens too often, consider "
            f"reducing sequence lengths or using a model with a larger context."
        )
        # for idx in reversed(indices_to_drop):  # reverse to avoid index shifting
        #     for key in tokenized:
        #         tokenized[key].pop(idx)
    return tokenized


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
)

# %%
# Explore tokenized dataset
# pprint(tokenized_dataset["train"])
# for key, value in tokenized_dataset["train"][0].items():
#     print(f"{key}: {value} ({type(value)})")


# %%
trl.DataCollatorForCompletionOnlyLM

#
# # Collate data
# class DataCollatorForCompletionOnlyLMPreComputed(transformers.DataCollatorForLanguageModeling):
#     """
#     Inspired by trl.DataCollatorForCompletionOnlyLM, but use pre-computed generation mask rather
#     than searching the input for instruction/response templates.
#     """
#
#     def __init__(self, *args, **kwargs):
#         if "mlm" in kwargs:
#             kwargs.pop("mlm")
#             logger.warning(
#                 "The `mlm` argument is not needed and will be ignored. "
#                 "This data collator is not designed for masked language modeling."
#             )
#         super().__init__(*args, mlm=False, **kwargs)
#
#         self.ignore_index = -100
#         # -100 seems to be hardcoded in some places of transformers for ignored tokens, e.g. in
#         # transformers.DataCollatorForLanguageModeling.torch_call (inherited by this class)
#
#     def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
#         def get_examples_attribute(attr: str | list[str]):
#             if isinstance(attr, str):
#                 return [e[attr] for e in examples]
#             else:
#                 return [{a: e[a] for a in attr} for e in examples]
#
#         input_ids = get_examples_attribute("input_ids")
#         batch = super().torch_call(input_ids)
#         batch_size, seq_length = batch["input_ids"].shape
#
#         # Manual management of assistant_masks as they are not correctly padded by the parent class
#         assistant_masks = torch.nn.utils.rnn.pad_sequence(
#             map(torch.BoolTensor, get_examples_attribute("assistant_masks")),
#             batch_first=True,
#             padding_value=False,
#         )
#
#         # Some context length checks
#         context_length_checker: torch.LongTensor = assistant_masks.sum(dim=1)
#         assert context_length_checker.shape == (batch_size,)
#         num_no_assistant = len(context_length_checker) - context_length_checker.count_nonzero()
#         num_full_assistant = (context_length_checker == seq_length).count_nonzero()
#         if num_no_assistant > 0:
#             logger.warning(
#                 f"{num_no_assistant} samples have no assistant tokens at all. "
#                 f"If this happens too often, consider increasing the max_seq_length."
#             )
#         if num_full_assistant > 0:
#             logger.warning(
#                 f"{num_full_assistant} samples have the full sequence as assistant tokens. "
#                 f"If this happens too often, consider increasing the max_seq_length."
#             )
#
#         # Mask assistant tokens
#         batch["labels"][~assistant_masks] = self.ignore_index
#         return batch


# data_collator = DataCollatorForCompletionOnlyLMPreComputed(
#     tokenizer=tokenizer,
#     # pad_to_multiple_of=8,
# )

data_collator = trl.DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="[/INST]",
    instruction_template="[INST]",
)

# %%
# Test data collator
for key, value in tokenized_dataset["train"][1000].items():
    value = value if not isinstance(value, torch.Tensor) else value.shape
    value = value if not isinstance(value, list) else f"list[{len(value)}]"
    print(f"{key} {str(type(value))}: {value}")
batch = tokenized_dataset["train"].batch(2)[0]
# print(batch.keys())
# print(torch.Tensor(batch["input_ids"]).shape)
collated = data_collator([{"input_ids": torch.LongTensor(batch["input_ids"][0])}])
# print(collated)

if False:
    s = """[INST]What does Big Data really do with our data?[/INST]"Big Data" isn't an entity so it does nothing with our data. "Big Data" refers to the concept of using large amounts of data to draw conclusions or direct action. Some company might collect "Big Data" about their customers and then try to draw conclusions about their customer base, such as what products to carry or advertise. Similarly selling that data to other companies would be profitable, and buying such data could in turn help grow a business."""
    st = tokenizer.encode(s, return_tensors="pt")
    print(st)
    print(tokenizer.tokenize(s))
    print(data_collator(st))

# %%
# Load model (if not already loaded, only useful when cell is re-run)
if False:
    try:
        model
    except NameError:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id)

# %%
if False:
    with torch.no_grad():
        print(model(torch.LongTensor(batch["input_ids"])))

# %%
# Test longest sample
if False:
    longest_sample = max(dataset["train"], key=lambda x: len(x["answers.text"][0]))
    sample_conversation = [
        # fmt: off
        {"type": "input", "content": longest_sample["title"]},
        {"type": "output", "content": longest_sample["answers.text"][0]},
        # fmt: on
    ]
    sample_conversation_tokenized = tokenizer.apply_chat_template(
        sample_conversation,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        return_tensors="pt",
    )
    sample_conversation_tokenized["input_ids"] = sample_conversation_tokenized["input_ids"]
    print_pretty_templated_with_mask(tokenizer, sample_conversation)
    print(len(sample_conversation_tokenized["input_ids"][0]))
    print(model.config.max_position_embeddings)
    # Raises an error if len(input) > max_position_embeddings
    print(model(sample_conversation_tokenized["input_ids"]))

if False:
    with torch.no_grad():
        input_ids = torch.ones((1, 2048), dtype=torch.long) * 2
        # print(model(input_ids))
        labels = torch.ones((1, 2048), dtype=torch.long) * 2
        labels[:, :5] = -100
        print(
            torch.nn.functional.cross_entropy(
                model(input_ids).logits.view(-1, model.config.vocab_size), labels.view(-1)
            )
        )
        labels[:, :50] = -100
        print(
            torch.nn.functional.cross_entropy(
                model(input_ids).logits.view(-1, model.config.vocab_size), labels.view(-1)
            )
        )
        labels[:, :500] = -100
        print(
            torch.nn.functional.cross_entropy(
                model(input_ids).logits.view(-1, model.config.vocab_size), labels.view(-1)
            )
        )
        labels[:, :] = -100
        print(
            torch.nn.functional.cross_entropy(
                model(input_ids).logits.view(-1, model.config.vocab_size), labels.view(-1)
            )
        )


# %%
# Prepare the training configuration
training_args = SFTConfig(
    max_seq_length=max_context_length,
    output_dir="/tmp",
    # remove_unused_columns=False,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    report_to="wandb",
)
trainer = SFTTrainer(
    model_id,
    train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    args=training_args,
    peft_config=peft.LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.1,
        inference_mode=False,
    ),
    # deal with quantization
    model_init_kwargs=dict(
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_storage="bfloat16",
        )
    ),
)

# %%
# Train the model
trainer.train()
