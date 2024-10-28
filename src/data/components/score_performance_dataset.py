from dataclasses import dataclass
from pathlib import Path
from typing import Any

import music21 as m21
import symusic
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from utils.abc_xml_conversion import conversion


@dataclass
class ScorePerformanceEntry:
    xml_score: Path
    midi_performance: Path
    midi_score_annotations: Path = None
    midi_score: Path = None
    composer: str = None
    title: str = None
    split: str = None

    def get_score_as_musicxml(self) -> str:
        """
        Return the MusicXML file as a string.
        """
        with open(self.xml_score, "r") as f:
            return f.read()

    def get_score_as_abc(self):
        """
        Return the MusicXML file converted into ABC format.
        """
        abc = conversion.xmlfile2abc(self.xml_score)
        return abc

    def get_score_as_music21(self):
        """
        Return the MusicXML file as a music21 score.
        """
        score = m21.converter.parse(self.xml_score)
        return score

    def get_midi_performance_as_symusic(self, ttype: str = "second"):
        """
        Return the MIDI performance as a symusic object.

        :param ttype: The time type to use. Can be "tick", "quarter" or "second".
        """
        midi = symusic.Score(self.midi_performance, ttype=ttype)
        return midi


class ScorePerformanceDataset(Dataset):
    """Dataset for a dataset of pairs of MusicXML scores and MIDI performances."""

    def __init__(self, entries: list[ScorePerformanceEntry]) -> None:
        """Initialize a dataset of pairs of MusicXML scores and MIDI performances.

        :param entries: A list of ScorePerformanceEntry objects.
        """
        self.entries = entries

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        :return: The number of samples in the dataset.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> ScorePerformanceEntry:
        """Return a sample from the dataset.

        :param idx: The index of the sample to return.
        :return: The sample at the given index.
        """
        return self.entries[idx]


class ScorePerformanceAsTextDataset(ScorePerformanceDataset):
    def __init__(
        self,
        entries: list[ScorePerformanceEntry],
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Initialize a dataset of pairs of MusicXML scores and MIDI performances, but in a textual
        representation suited for language models.

        :param entries: A list of ScorePerformanceEntry objects.
        :param tokenizer: The tokenizer to use.
        """
        super().__init__(entries)
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Any:
        """Return a sample from the dataset.

        :param idx: The index of the sample to return.
        :return: A dictionary with the following keys:
            - `midi_performance_as_text`: The MIDI performance as text (for debugging purposes).
            - `abc_score_as_text`: The ABC score as text (for debugging purposes).
            - `tokenized_conversation`: The tokenized conversation between the user and the assistant.
            - `assistant_mask`: As mask indicating the assistant tokens in the tokenized conversation (1's).
        """
        entry = super().__getitem__(idx)
        midi_as_text = "\n".join(
            [
                f"Note(pitch={note.pitch}, start={round(note.start, 3)}, duration={round(note.duration, 3)})"  # Velocity unused for now
                for note in entry.get_midi_performance_as_symusic(ttype="second").tracks[0].notes
            ]
        )
        score_as_text = entry.get_score_as_abc()
        conversation = [
            dict(role="user", content=midi_as_text),
            dict(role="assistant", content=score_as_text),
        ]
        tokenized_conversation = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_dict=True,
            return_assistant_tokens_mask=True,  # currently broken (see https://github.com/huggingface/transformers/issues/33091), see PR (https://github.com/huggingface/tokenizers/pull/1640)
        )

        return {
            "midi_performance_as_text": midi_as_text,
            "abc_score_as_text": entry.get_score_as_abc(),
            "tokenized_conversation": tokenized_conversation["input_ids"],
            "assistant_masks": tokenized_conversation["assistant_masks"],
        }
