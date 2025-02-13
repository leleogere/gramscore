def default_voice_to_staff(voice_idx: int, one_based: bool = False) -> int:
    """
    Default voice to staff mapping. Following MusicXML/MuseScore standards, the four first voices
    are mapped to the first staff, the four next voices to the second staff, and so on.

    :param voice_idx: Voice index (0-based or 1-based)
    :param one_based: Whether the voice and staff indices are 1-based (True) or 0-based (False)

    :return: Staff index (0 or 1-based depending on the parameter `one_based`)
    """
    if one_based:
        voice_idx -= 1
    return voice_idx // 4 + (1 if one_based else 0)
