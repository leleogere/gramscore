import peft
import trl
import datasets
from loguru import logger
from trl import SFTConfig, SFTTrainer
import transformers

from data.components.patched_tokenizer import PatchedTokenizer

transformers.set_seed(10)

TEXT_DATASET = (
    "/home/gerel/Documents/datasets/nasap-dataset_chunks=32_overlap=True_min-ratio=0.6_TEXT"
).removesuffix("/")


# %%
# Load dataset
dataset = datasets.DatasetDict()
for split in ["train", "validation", "test"]:
    print(f"Loading {split} split...")
    dataset[split] = datasets.load_dataset(
        "text",
        data_dir=f"{TEXT_DATASET}/{split}",
        split=split,
        sample_by="document",
    )


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


# %%
# Prepare dataset
SEPARATOR = "\n\n" + "#" * 60 + "\n" + "#" * 60 + "\n\n"


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
    remove_columns=dataset["train"].column_names,
)

data_collator = trl.DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="[/INST]",
    instruction_template="[INST]",
)


# %%
# Prepare the training configuration
training_args = SFTConfig(
    max_seq_length=max_context_length,
    output_dir="/tmp",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    report_to="wandb",
)
trainer = SFTTrainer(
    model_id,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
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
