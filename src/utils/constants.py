SUCCESS_FILENAME = ".success"
PADDING_VALUE = -100

# ======================================================== #
# Variables from MIDI2ScoreTransformer for fair comparison #
# ======================================================== #
# Not parsable by music21
SKIP = set(["{ASAP}/Glinka/The_Lark/Denisova10M.mid", "{ASAP}/Glinka/The_Lark/Kleisen07M.mid"])

# Not aligned correctly or other issues
# fmt: off
TO_IGNORE_INDICES = [
    152, 153, 154, 165, 166, 179, 180, 181, 332, 333, 334, 335, 349, 350,
    351, 418, 419, 420, 426, 428, 429, 430, 472, 473, 474, 489, 490, 491,
    516, 517, 518, 519, 520, 521, 522, 540, 541, 560, 609, 774, 798, 799,
    800, 801, 802, 803, 819, 920, 921, 935, 936, 937, 938, 939, 940, 941,
    979, 980, 981, 997, 998, 999, 1012, 1013, 1014, 1017, 1018
]
# fmt: on

# To keep eval consistent, we hardcode test piece ids here.
TEST_PIECE_IDS = [15, 78, 159, 172, 254, 288, 322, 374, 395, 399, 411, 418, 452, 478]
