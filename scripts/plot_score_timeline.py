import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import partitura as pt
import partitura.score
from matplotlib.patches import Polygon
from tqdm import tqdm


DEFAULT_DURATION = 100
TRIANGLE_WIDTH = 5
RECTANGLE_HEIGHT = 0.4
WIDTH_FACTOR = 0.6


def compute_lanes(events):
    lanes = []
    positions = []

    for event in events:
        start_t = event.start.t if event.start else event.end.t
        if event.end:
            end_t = event.end.t
            open_end = False
        else:
            end_t = start_t + DEFAULT_DURATION
            open_end = True
        if start_t > end_t:
            start_t, end_t = end_t, start_t

        for i, lane in enumerate(lanes):
            if all(end_t <= e[0] or start_t >= e[1] for e in lane):
                lane.append((start_t, end_t))
                positions.append((event, i, start_t, end_t, open_end))
                break
        else:
            lanes.append([(start_t, end_t)])
            positions.append((event, len(lanes) - 1, start_t, end_t, open_end))

    return positions, len(lanes)


def plot_timeline(events, filename="timeline.pdf"):
    positions, n_lanes = compute_lanes(events)
    times = [t * WIDTH_FACTOR for _, _, start, end, _ in positions for t in (start, end)]
    t_min, t_max = min(times), max(times)

    types = sorted(set(type(e).__name__ for e, *_ in positions))
    cmap = cm.get_cmap('tab20', len(types))
    color_map = {t: cmap(i) for i, t in enumerate(types)}

    fig_height = n_lanes * RECTANGLE_HEIGHT
    fig, ax = plt.subplots(figsize=((t_max - t_min + 2 * TRIANGLE_WIDTH) / 5, fig_height))
    ax.set_xlim(t_min - 1, t_max + TRIANGLE_WIDTH + 1)
    ax.set_ylim(-1, n_lanes)
    ax.set_xlabel("Time")
    ax.yaxis.set_visible(False)

    for event, y, start_t, end_t, open_end in positions:
        color = color_map[type(event).__name__]

        if open_end:
            points = [
                (start_t, y - RECTANGLE_HEIGHT),
                (end_t, y - RECTANGLE_HEIGHT),
                (end_t + TRIANGLE_WIDTH, y),
                (end_t, y + RECTANGLE_HEIGHT),
                (start_t, y + RECTANGLE_HEIGHT),
            ]
        else:
            points = [
                (start_t, y - RECTANGLE_HEIGHT),
                (end_t, y - RECTANGLE_HEIGHT),
                (end_t, y + RECTANGLE_HEIGHT),
                (start_t, y + RECTANGLE_HEIGHT),
            ]

        polygon = Polygon(points, closed=True, facecolor=color, edgecolor='black')
        ax.add_patch(polygon)
        ax.text(
            start_t + (end_t - start_t) / 2, y, str(event), ha='center', va='center', fontsize=8
        )

    fig.tight_layout()
    fig.savefig(filename, dpi=50)
    plt.close(fig)


if __name__ == '__main__':
    score_path = Path(
        "~/Documents/datasets/asap-dataset/Ravel/Miroirs/3_Une_Barque/xml_score.musicxml"
    ).expanduser()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(score_path)
        assert len(score.parts) == 1
        score = score.parts[0]

    events = [
        e
        for e in score.iter_all()
        if not isinstance(
            e,
            (
                pt.score.Page,
                pt.score.System,
                pt.score.Direction,
            ),
        )
    ][:10000]
    plot_timeline(events, filename="/tmp/plot.png")
