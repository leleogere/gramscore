from collections.abc import Callable


class LinearRamp:
    """Generic linear scheduler"""

    def __init__(
        self,
        coefficient: float,
        x_offset: float | None = None,
        y_offset: float | None = None,  # exclusive with x_offset
        clamp_min: float | None = None,
        clamp_max: float | None = None,
        cast_type: Callable | None = None,
    ):
        if x_offset is not None and y_offset is not None:
            raise ValueError("Cannot specify both x_offset and y_offset")

        self.coefficient = coefficient
        if x_offset is not None:
            y_offset = -coefficient * x_offset
        self.y_offset = y_offset if y_offset is not None else 0.0
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        if isinstance(cast_type, str):
            cast_type = eval(cast_type)  # for hydra config
        self.cast_type = cast_type

    def __call__(self, step: int):
        value = self.coefficient * step + self.y_offset
        if self.clamp_min is not None and value < self.clamp_min:
            value = self.clamp_min
        if self.clamp_max is not None and value > self.clamp_max:
            value = self.clamp_max
        if self.cast_type is not None:
            value = self.cast_type(value)
        return value


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    scheduler = LinearRamp(
        coefficient=500 / 100_000,
        clamp_min=50,
        clamp_max=500,
    )

    steps = np.linspace(0, 120_000, 10_000)
    values = [scheduler(s) for s in steps]

    plt.plot(steps, values)
    plt.xlabel("Training Steps")
    plt.ylabel("Values")
    plt.grid()
    plt.show()
