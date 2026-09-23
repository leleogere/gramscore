"""
Find the maximum number of voices per measure in ASAP
"""

from black.trans import defaultdict
from lxml import etree
from pathlib import Path

asap_path = Path("/home/gerel/Documents/datasets/asap-dataset")
score_paths = asap_path.rglob("xml_score.musicxml")

max_voices = 0
max_voice_to_path = defaultdict(lambda: defaultdict(list))

for score_path in score_paths:
    current_max_voice = 0
    tree = etree.parse(score_path)
    root = tree.getroot()

    for element in root.findall(".//measure"):
        measure_idx = element.get("number")
        voices = set()
        for note in element.findall(".//note"):
            voice = note.find("voice").text
            voices.add(voice)
        if len(voices) > current_max_voice:
            if current_max_voice in max_voice_to_path:
                max_voice_to_path[current_max_voice].pop(score_path)
                if len(max_voice_to_path[current_max_voice]) == 0:
                    max_voice_to_path.pop(current_max_voice)
            current_max_voice = len(voices)
            max_voice_to_path[current_max_voice][score_path] = [measure_idx]
            # max_voice_to_path[current_max_voice][score_path].append(measure_idx)
        elif len(voices) == current_max_voice:
            max_voice_to_path[current_max_voice][score_path].append(measure_idx)

# Sort by maximum number of voices
max_voice_to_path = dict(sorted(max_voice_to_path.items(), key=lambda x: x[0]))
max_voices = max(max_voice_to_path.keys())

for voice, paths in max_voice_to_path.items():
    num_scores = len(paths)
    num_measures = sum(len(measures) for measures in paths.values())
    print(f"Max voices: {voice} (found in {num_scores} scores, {num_measures} measures)")
for path, measures in max_voice_to_path[max_voices].items():
    print(f"  * {path} (measures {', '.join(measures)})")

"""
Max voices: 2 (found in 4 scores, 211 measures)
Max voices: 3 (found in 25 scores, 669 measures)
Max voices: 4 (found in 103 scores, 1682 measures)
Max voices: 5 (found in 76 scores, 527 measures)
Max voices: 6 (found in 21 scores, 127 measures)
Max voices: 7 (found in 2 scores, 4 measures)
Max voices: 8 (found in 4 scores, 10 measures)
  * Beethoven/Piano_Sonatas/14-3/xml_score.musicxml (measures 163, 164, 165, 166)
  * Beethoven/Piano_Sonatas/23-1/xml_score.musicxml (measures 123)
  * Beethoven/Piano_Sonatas/3-1/xml_score.musicxml (measures 222, 224, 226)
  * Beethoven/Piano_Sonatas/18-1/xml_score.musicxml (measures 136, 137)
"""
