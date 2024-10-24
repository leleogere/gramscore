"""
This tokenizer patch the chat template of the codestral tokenizer to include the "generation" tags.
It won't be necessary once the PR https://huggingface.co/mistralai/Codestral-22B-v0.1/discussions/52
is merged and released.
"""

import transformers

INITIAL_TEMPLATE = "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.last and system_message is defined %}\n            {{- '[INST] ' + system_message + '\\n\\n' + message['content'] + '[/INST]' }}\n        {%- else %}\n            {{- '[INST] ' + message['content'] + '[/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"
PATCHED_TEMPLATE = "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.last and system_message is defined %}\n            {{- '[INST] ' + system_message + '\\n\\n' + message['content'] + '[/INST]' }}\n        {%- else %}\n            {{- '[INST] ' + message['content'] + '[/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {%- generation %}\n            {{- ' ' + message['content'] + eos_token}}\n        {%- endgeneration %}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"


class PatchedTokenizer(transformers.AutoTokenizer):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        self = super().from_pretrained(*args, **kwargs)
        if self.name_or_path == 'mistralai/Codestral-22B-v0.1':
            if self.chat_template != INITIAL_TEMPLATE:
                raise ValueError(
                    "The chat templates of this tokenizer does not correspond to the Codestral one. "
                    "Update the patch, or if PR #52 has been merged remove it completely: "
                    "https://huggingface.co/mistralai/Codestral-22B-v0.1/discussions/52"
                )
            self.chat_template = PATCHED_TEMPLATE
        return self
