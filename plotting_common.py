from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
MPLCONFIG_DIR = CACHE_DIR / "matplotlib"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib
import numpy as np
from matplotlib.ticker import LogLocator


def _in_ipykernel() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and hasattr(shell, "kernel")


if not _in_ipykernel() and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

TEXTWIDTH_PT = 510.0
COLUMNWIDTH_PT = 246.0
X_MINOR = LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)

SAVE_PLOT_KWARGS = {"bbox_inches": "tight", "transparent": False}

def configure_matplotlib(fontsize: int = 10) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": fontsize,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "legend.title_fontsize": fontsize,
            "legend.borderpad": 0.3,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 1.0,
            "legend.columnspacing": 1.0,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "lines.linewidth": 2,
            "lines.color": "k",
            "figure.titlesize": fontsize,
            "savefig.format": "pdf",
            "savefig.dpi": 72.27,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    stem: str,
    *,
    fig=None,
    show: bool | None = None,
    close: bool = True,
) -> None:
    if fig is None:
        fig = plt.gcf()
    if show is None:
        show = _in_ipykernel() or bool(plt.isinteractive())

    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(pdf_path, **SAVE_PLOT_KWARGS)

    if show:
        plt.show()

    if close:
        plt.close(fig)


def load_npz_dataset(name: str) -> dict[str, np.ndarray | float | int]:
    path = DATA_DIR / f"{name}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: archive[key].item() if archive[key].shape == () else archive[key].copy()
            for key in archive.files
        }


def set_size(width_pt: float, fraction: float = 1, subplots: tuple[int, int] = (1, 1), height_factor: float = 1.0) -> tuple[float, float]:
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in * height_factor)


def lighten_color(color, amount: float = 0.5):
    import colorsys
    import matplotlib.colors as mc

    try:
        resolved_color = mc.cnames[color]
    except KeyError:
        resolved_color = color
    hue, lightness, saturation = colorsys.rgb_to_hls(*mc.to_rgb(resolved_color))
    return colorsys.hls_to_rgb(hue, 1 - amount * (1 - lightness), saturation)


def moving_time_integral(time: np.ndarray, observable: np.ndarray, max_points: int = 100) -> np.ndarray:
    averaged = np.zeros_like(observable, dtype=float)

    for index in range(len(time)):
        start = max(0, index - max_points + 1)
        t_segment = time[start : index + 1]
        observable_segment = observable[start : index + 1]
        dt = t_segment[-1] - t_segment[0]
        if dt != 0:
            averaged[index] = np.trapz(observable_segment, t_segment) / dt
        else:
            averaged[index] = observable_segment[0]

    return averaged


def moving_time_integral_centered(
    time: np.ndarray, observable: np.ndarray, max_points: int = 101
) -> np.ndarray:
    n_points = len(time)
    if n_points != len(observable):
        raise ValueError("time and observable must have the same length")

    if max_points % 2 == 0:
        max_points += 1

    half_window = max_points // 2
    averaged = np.zeros_like(observable, dtype=float)

    for index in range(n_points):
        radius = min(half_window, index, n_points - 1 - index)
        start = index - radius
        end = index + radius + 1
        t_segment = time[start:end]
        observable_segment = observable[start:end]
        dt = t_segment[-1] - t_segment[0]
        if dt != 0:
            averaged[index] = np.trapz(observable_segment, t_segment) / dt
        else:
            averaged[index] = observable_segment[0]

    return averaged
