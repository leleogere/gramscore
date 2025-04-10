from datetime import datetime
from pathlib import Path
import copy

import datasets
import trl
from loguru import logger
from trl import SFTConfig, SFTTrainer
import unsloth
import transformers
import torch
import numpy as np

MAX_SEQ_LENGTH = 32768  # Supports automatic RoPE Scaling, so choose any number
# TEXT_DATASET = (
#     "/local/DEEPLEARNING/EPR/nasap-dataset_chunks=32_overlap=True_min-ratio=0.6_TEXT"
# ).removesuffix("/")
TEXT_DATASET = (
    "/local/DEEPLEARNING/EPR/nasap-dataset_chunks=16_overlap=True_min-ratio=0.6_TEXT"
).removesuffix("/")
TEXT_DATASET = Path(TEXT_DATASET)

cached_dataset = TEXT_DATASET.parent / f"{TEXT_DATASET.name}_preprocessed"

EVAL_STEPS = 100
SAVE_STEPS = 100
LOGGING_STEPS = 10
EPOCHS = 10

# %% Load model
model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype="bfloat16",  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False
)
print(f"{tokenizer.pad_token = }")

# %% Setup tokenizer
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

# %% Do model patching and add fast LoRA weights
model = unsloth.FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=[
        "embed_tokens",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Dropout = 0 is currently optimized
    bias="none",  # Bias = "none" is currently optimized
    # use_gradient_checkpointing=True,
    random_state=10,
)

# %% Prepare the dataset
if cached_dataset.exists():
    tokenized_dataset = datasets.load_from_disk(cached_dataset)
    print(f"Dataset loaded from disk at {cached_dataset}")
else:
    dataset = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        print(f"Loading {split} split...")
        dataset[split] = datasets.load_dataset(
            "text",
            data_dir=str(TEXT_DATASET / split),
            sample_by="document",
        )[
            "train"
        ]  # Needed to flatten the dataset
    dataset["validation"] = dataset["validation"].shuffle(seed=10).take(200)  # Reduce validation set for faster training


    # overfit_single_batch = False
    # fake_dataset = False
    # if overfit_single_batch:
    #     print("!!! CURRENTLY OVERFITTING A SINGLE BATCH !!!")
    #     if fake_dataset:
    #         print("!!! USING FAKE DATASET !!!")
    #         # fake_data = "Note(1, 2, 3)\nNote(2, 8, 1)\nNote(4, 2, 7)" + "\n\n" + "#" * 60 + "\n" + "#" * 60 + "\n\n" + "ABCBHADBG"
    #         fake_data = """Note(pitch=68, start=0, duration=3)
    # Note(pitch=72, start=27, duration=3)
    # Note(pitch=75, start=52, duration=3)
    # Note(pitch=80, start=76, duration=4)
    # Note(pitch=84, start=101, duration=31)
    # Note(pitch=60, start=128, duration=4)
    # Note(pitch=63, start=128, duration=3)
    # Note(pitch=56, start=128, duration=3)
    # Note(pitch=82, start=135, duration=5)
    # Note(pitch=80, start=139, duration=4)
    # Note(pitch=79, start=145, duration=6)
    # Note(pitch=80, start=150, duration=3)
    # Note(pitch=63, start=153, duration=4)
    # Note(pitch=56, start=153, duration=4)
    # Note(pitch=60, start=154, duration=3)
    # Note(pitch=56, start=178, duration=4)
    # Note(pitch=60, start=178, duration=4)
    # Note(pitch=63, start=179, duration=3)
    # Note(pitch=73, start=198, duration=10)
    # Note(pitch=85, start=204, duration=38)
    # Note(pitch=63, start=230, duration=3)
    # Note(pitch=58, start=230, duration=3)
    # Note(pitch=61, start=231, duration=3)
    # Note(pitch=55, start=231, duration=3)
    # Note(pitch=84, start=240, duration=6)
    # Note(pitch=82, start=245, duration=4)
    # Note(pitch=81, start=249, duration=8)
    # Note(pitch=82, start=256, duration=2)
    # Note(pitch=63, start=257, duration=3)
    # Note(pitch=55, start=258, duration=4)
    # Note(pitch=58, start=258, duration=3)
    # Note(pitch=61, start=259, duration=3)
    # Note(pitch=61, start=282, duration=3)
    # Note(pitch=58, start=283, duration=3)
    # Note(pitch=55, start=283, duration=4)
    # Note(pitch=63, start=283, duration=3)
    # Note(pitch=70, start=307, duration=3)
    # Note(pitch=73, start=335, duration=5)
    # Note(pitch=79, start=359, duration=4)
    # Note(pitch=82, start=383, duration=4)
    # Note(pitch=73, start=406, duration=5)
    # Note(pitch=85, start=413, duration=30)
    # Note(pitch=58, start=435, duration=3)
    # Note(pitch=64, start=436, duration=3)
    # Note(pitch=55, start=436, duration=3)
    # Note(pitch=61, start=437, duration=4)
    # Note(pitch=84, start=445, duration=6)
    # Note(pitch=82, start=449, duration=4)
    # Note(pitch=81, start=455, duration=6)
    # Note(pitch=82, start=461, duration=3)
    # Note(pitch=64, start=461, duration=3)
    # Note(pitch=58, start=462, duration=3)
    # Note(pitch=61, start=462, duration=3)
    # Note(pitch=55, start=464, duration=5)
    # Note(pitch=55, start=486, duration=4)
    # Note(pitch=58, start=486, duration=4)
    # Note(pitch=64, start=487, duration=3)
    # Note(pitch=61, start=487, duration=3)
    # Note(pitch=73, start=507, duration=12)
    # Note(pitch=85, start=514, duration=37)
    # Note(pitch=64, start=538, duration=3)
    # Note(pitch=54, start=538, duration=4)
    # Note(pitch=58, start=539, duration=4)
    # Note(pitch=61, start=539, duration=2)
    # Note(pitch=84, start=549, duration=7)
    # Note(pitch=82, start=553, duration=5)
    # Note(pitch=81, start=559, duration=7)
    # Note(pitch=64, start=564, duration=5)
    # Note(pitch=54, start=565, duration=6)
    # Note(pitch=61, start=565, duration=4)
    # Note(pitch=82, start=565, duration=4)
    # Note(pitch=58, start=565, duration=5)
    # Note(pitch=54, start=589, duration=9)
    # Note(pitch=64, start=589, duration=6)
    # Note(pitch=61, start=590, duration=4)
    # Note(pitch=58, start=590, duration=7)
    # Note(pitch=53, start=619, duration=24)
    # Note(pitch=81, start=621, duration=43)
    # Note(pitch=65, start=638, duration=13)
    # Note(pitch=53, start=649, duration=14)
    # Note(pitch=65, start=662, duration=12)
    # Note(pitch=53, start=675, duration=13)
    # Note(pitch=65, start=691, duration=13)
    # Note(pitch=78, start=705, duration=35)
    # Note(pitch=53, start=706, duration=11)
    # Note(pitch=65, start=721, duration=13)
    # Note(pitch=77, start=736, duration=47)
    # Note(pitch=53, start=736, duration=12)
    # Note(pitch=65, start=749, duration=10)
    # Note(pitch=75, start=761, duration=30)
    # Note(pitch=53, start=762, duration=7)
    # Note(pitch=65, start=773, duration=10)
    # Note(pitch=53, start=784, duration=9)
    # Note(pitch=72, start=785, duration=23)
    # Note(pitch=65, start=796, duration=7)
    # Note(pitch=53, start=809, duration=9)
    # Note(pitch=69, start=809, duration=18)
    # Note(pitch=65, start=819, duration=12)
    # Note(pitch=63, start=834, duration=54)
    # Note(pitch=66, start=834, duration=41)
    # Note(pitch=53, start=835, duration=7)
    # Note(pitch=53, start=866, duration=4)
    # Note(pitch=65, start=876, duration=9)
    # Note(pitch=61, start=886, duration=12)
    # Note(pitch=70, start=886, duration=2)
    # Note(pitch=53, start=887, duration=9)
    # Note(pitch=65, start=900, duration=12)
    # Note(pitch=78, start=911, duration=31)
    # Note(pitch=53, start=913, duration=7)
    # Note(pitch=65, start=924, duration=13)
    # Note(pitch=77, start=937, duration=45)
    # Note(pitch=53, start=939, duration=8)
    # Note(pitch=65, start=952, duration=9)
    # Note(pitch=75, start=963, duration=31)
    # Note(pitch=53, start=964, duration=8)
    # Note(pitch=65, start=976, duration=10)
    # Note(pitch=72, start=988, duration=23)
    # Note(pitch=53, start=988, duration=7)
    # Note(pitch=65, start=1000, duration=11)
    # Note(pitch=69, start=1012, duration=18)
    # Note(pitch=53, start=1014, duration=8)
    # Note(pitch=65, start=1024, duration=14)
    # Note(pitch=66, start=1037, duration=41)
    # Note(pitch=63, start=1037, duration=55)
    # Note(pitch=53, start=1038, duration=4)
    # Note(pitch=53, start=1068, duration=5)
    # Note(pitch=65, start=1080, duration=9)
    # Note(pitch=70, start=1089, duration=2)
    # Note(pitch=61, start=1090, duration=11)
    # Note(pitch=53, start=1090, duration=8)
    # Note(pitch=65, start=1102, duration=12)
    # Note(pitch=78, start=1113, duration=33)
    # Note(pitch=53, start=1115, duration=11)
    # Note(pitch=65, start=1126, duration=12)
    # Note(pitch=53, start=1139, duration=13)
    # Note(pitch=77, start=1140, duration=38)
    # Note(pitch=65, start=1152, duration=11)
    # Note(pitch=53, start=1164, duration=6)
    # Note(pitch=75, start=1164, duration=29)
    # Note(pitch=65, start=1174, duration=12)
    # Note(pitch=63, start=1188, duration=9)
    # Note(pitch=72, start=1189, duration=24)
    # Note(pitch=65, start=1202, duration=11)
    # Note(pitch=69, start=1214, duration=6)
    # Note(pitch=63, start=1214, duration=6)
    # Note(pitch=65, start=1227, duration=12)
    # Note(pitch=61, start=1239, duration=7)
    #
    # ############################################################
    # ############################################################
    #
    # X:1
    # T:Title
    # %%score { ( 1 4 ) | ( 2 3 ) }
    # L:1/8
    # M:2/2
    # I:linebreak $
    # K:Ab
    # V:1 treble nm="Piano"
    # V:4 treble
    # L:1/4
    # V:2 bass
    # V:3 bass
    # L:1/4
    # V:1
    # !<(! .A2 .c2 .e2 .a2!<)! | (!>!c'3!>(! (3(1:1:3(3b/ a/ g/!>)! .a2) z2 | %2
    # !>(!{/d} (!>!d'3 (3(1:1:3(3c'/ b/ =a/!>)! .b2) z2 | .B2 .d2 .g2 .b2 | %4
    # !>(!{/d} (!>!d'3 (3(1:1:3(3c'/ b/ =a/!>)! .b2) z2 | %5
    # !<(!{/d} (!>!d'3 (3(1:1:3(3(1:1:3c'/ b/ =a/ .b2) z2!<)! | .=a2 z2 z2 (_g2 |!<(! f2 e2 c2 =A2!<)! | %8
    #  _G3!>(! F)!>)! .B2 _g2 |!<(! (f2 e2 c2 =A2!<)! |!>(! _G3!>)! F) .B2 _g2 |!>(! (f2 e2 c2 =A2)!>)! | %12
    #  z (=A c B) z!<(! (B d c) | z (c e d) z (=B =d!<)! c) | .=B2 z2 z2 (a2 |!<(! g2 f2 =d2 =B2!<)! | %16
    #  A3!>(! G)!>)! .c2 a2 |!<(! (g2 f2 =d2 =B2!<)! | A3!>(! G)!>)! .c2!p! F =D | %19
    # !<(! F =D F D F D F D!<)! | F!>(! =D F D!>)! E C!<(! E C | E C E C E C E C!<)! | %22
    #  E C!>(! E C D!>)! B, D B, |!<(! D B, D B, D B, D B,!<)! | D B, D B, D B, D B, | %25
    # !<(! C A, E A, E A, E A, | D A, D A, D A, D A, | D G, D G, D G, D G, | C G, C G, C G, C G, | %29
    # !f! A, F, C F, C F, C F, | D F, D F, D F, D F, | =D F, [FDA,] F, [FDA,] F, [FDA,] F,!<)! | %32
    # V:2
    #  z8 | z2 .[ECA,]2 .[ECA,]2 .[ECA,]2 | z2 .[EDB,G,]2 .[EDB,G,]2 .[EDB,G,]2 | z8 | %4
    #  z2 [=EDB,G,]2 [EDB,G,]2 [EDB,G,]2 | z2 [=EDB,_G,]2 [EDB,G,]2 [EDB,G,]2 | F, F F, F F, F F, F | %7
    #  F, F F, F F, F F, F | F,2 F,2 F, F F, F | F, F F, F F, F F, F | F,2 F,2 F, F F, F | %11
    #  F, F F, F E F E F | D F D F C F C F | B, F B, F A, ^F A, F | G, G G, G G, G G, G | %15
    #  G, G G, G G, G G, G | G,2 G,2 G, G G, G | G, G G, G G, G G, G | G,2 G,2 G,2 A,2 | %19
    #  (G,2 F,2 =D,2 =B,,2) | (A,,3 G,, .C,2) _G,2 | F,2 E,2 C,2 =A,,2 | _G,,3 F,, B,,2 (_F,2 | %23
    #  E,2 D,2 B,,2 G,,2 | _F,,2 E,,4) (E,2- | E,2 C,4 C,,2- | C,,2 D,,4) (D,2- | D,2 B,,4 B,,,2- | %28
    #  B,,,2 C,,4) (C,2- | C,2 A,,4 A,,,2) | z2 (B,,,4 B,,2) | z2 =B,,,4 =B,,2 | %32
    # V:3
    #  x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | (E2 .D) x | x4 | (E2 .D) x | x4 | x4 | x4 | x4 | x4 | %16
    #  (F2 .E) x | x4 | (F2 .E) x | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | %32
    # V:4
    #  x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | x4 | %19
    #  x4 | z (=B c) z | z4 | z (=A B) z | z4 | z2 G2 | z2 A2 | z2 F2 | z2 G2 | z2 =E2 | z2 F2 | z2 F2 | %31
    #  x4 | %32
    #
    # """
    #         fake_dataset = datasets.Dataset.from_list(mapping=[{"text": fake_data}])
    #         dataset["train"] = fake_dataset
    #         dataset["validation"] = fake_dataset
    #     else:
    #         dataset["train"] = dataset["train"].take(1)
    #         dataset["validation"] = dataset["train"].take(1)
    #     assert all(t == v for t, v in zip(dataset["train"]["text"], dataset["validation"]["text"]))
    #     print("NUM_TOKENS:", len(tokenizer.encode(dataset["train"]["text"][0])))
    #     EVAL_STEPS = 1
    #     LOGGING_STEPS = 1
    #     EPOCHS=100


    def preprocess_function(samples):
        sep = "\n\n" + "#" * 60 + "\n" + "#" * 60 + "\n\n"
        tokenized = tokenizer.apply_chat_template(
            [
                [
                    {"role": "user", "content": (splitted := text.split(sep))[0]},
                    {"role": "assistant", "content": splitted[1]},
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
        tokenized["to_drop"] = [len(ids) > MAX_SEQ_LENGTH for ids in tokenized["input_ids"]]
        num_to_drop = sum(tokenized["to_drop"])
        if num_to_drop > 0:
            logger.warning(
                f"{num_to_drop} samples of the batch are longer than {MAX_SEQ_LENGTH} and will be "
                f"dropped. If this happens too often, consider reducing sequence lengths or using a "
                f"model with a larger context."
            )
        # Add labels and mask user tokens
        tokenized["labels"] = [
            [
                token if is_assistant else -100
                for token, is_assistant in zip(input_ids, assistant_masks)
            ]
            for input_ids, assistant_masks in zip(tokenized["input_ids"], tokenized["assistant_masks"])
        ]
        return tokenized


    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=20,
    ).filter(lambda x: not x["to_drop"], num_proc=20)

    tokenized_dataset.save_to_disk(
        dataset_dict_path=cached_dataset,
        num_proc=20,
    )
    print(f"Dataset saved to disk at {cached_dataset}")

# exit()


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
            grad_min, grad_mean, grad_max = gradients.min(), gradients.mean(), gradients.max()
            print(f"{grad_min = }, {grad_mean = }, {grad_max = }")
        else:
            print("No gradients to inspect")
        # Get min, mean and max of the weights
        weights = [param for param in model.parameters()]
        if weights:
            weights = torch.cat([weight.flatten() for weight in weights])
            weights_min, weights_mean, weights_max = weights.min(), weights.mean(), weights.max()
            print(f"{weights_min = }, {weights_mean = }, {weights_max = }")
        else:
            print("No weights to inspect")

    def on_step_begin(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        step = state.global_step
        print(f"Step {step} begin")
        train_dl = kwargs['train_dataloader']
        for i, batch in enumerate(train_dl):
            if i < step % len(train_dl):
                continue
            keys = list(batch.keys())
            for key in keys:
                value = batch[key][0]
                if isinstance(value, torch.Tensor):
                    value = value.cpu().tolist()
                if isinstance(value, list):
                    value = f"[{', '.join(map(str, value[:10]))}, ..., {', '.join(map(str, value[-10:]))}]<{len(value)}>"
                print(f"{key = }, {value = }")
            break


# %% Prepare DataCollator
data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding="longest",
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)


# %% Prepare the training configuration
output_dir = Path(f"./outputs/{datetime.now().strftime('%Y%m%d_%H%M')}")
output_dir.mkdir(parents=True, exist_ok=False)
training_args = SFTConfig(
    max_seq_length=MAX_SEQ_LENGTH,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,
    # gradient_checkpointing=True,
    # optim="adam",
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    report_to="wandb",
    output_dir=str(output_dir),
    logging_dir=str(output_dir / "logs"),
    group_by_length=True,
    bf16=True,
    seed=10,
)
trainer = SFTTrainer(
    model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    # data_collator=transformers.DefaultDataCollator(return_tensors="pt"),
    data_collator=data_collator,
    args=training_args,
    # callbacks=[InspectionCallback(), transformers.trainer_callback.PrinterCallback()],
)
# TODO: use unsloth trainer to decouple lm_head and embedding LR from rest of the model?

# %% Train the model
trainer_stats = unsloth.unsloth_train(trainer)  # Fixes gradient accumulation bug (unsloth #1134)
print(trainer_stats)
