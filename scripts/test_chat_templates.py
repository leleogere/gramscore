import transformers


# %% Test function
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
        # add_generation_prompt=True,
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


# %% Load tokenizer
name = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
tokenizer = transformers.AutoTokenizer.from_pretrained(name)

# %% Setup chat template
chat_template = """
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "assistant") %}
        {{- '<|im_start|>' + message.role + '\\n' }}
        {%- if message.role == "assistant" %}
            {%- generation %}
            {{- message.content + '<|im_end|>' }}
            {%- endgeneration %}
            {{- '\\n' }}
        {%- else %}
            {{- message.content + '<|im_end|>\\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Only "user" and "assistant" message types accepted.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}
""".strip()
tokenizer.chat_template = chat_template


# %% Test the template
sample = {
    "messages": [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you."},
    ]
}
print_pretty_templated_with_mask(tokenizer, sample["messages"])


# %% Test with model
model = transformers.AutoModel.from_pretrained(name)
print(model)
