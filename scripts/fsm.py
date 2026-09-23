import random


class Voice:
    def __init__(self, final_length: float = 4):
        self.current_length = 0.0
        self.final_length = final_length

    def increment(self, increment: float):
        new_length = self.current_length + increment
        if new_length > self.final_length:
            raise ValueError("Voice has reached final length")
        self.current_length = new_length

    @property
    def is_started(self):
        return self.current_length > 0

    @property
    def is_finished(self):
        return self.current_length == self.final_length

    @property
    def is_active(self):
        return self.is_started and not self.is_finished

    @property
    def remaining_length(self):
        return self.final_length - self.current_length


class Generator:
    def __init__(self, num_voices: int = 3, voice_length: float = 4):
        self.voices = [Voice(voice_length) for _ in range(num_voices)]

    def random_walk(self):
        while not all(voice.is_finished for voice in self.voices):
            voice = self.get_random_voice()
            increment = round(random.random() * voice.remaining_length * 4) / 4
            voice.increment(increment)
            print([voice.current_length for voice in self.voices])

    def get_random_voice(self):
        return random.choice([voice for voice in self.voices if not voice.is_finished])


if __name__ == '__main__':
    generator = Generator()
    generator.random_walk()
    print([voice.current_length for voice in generator.voices])
