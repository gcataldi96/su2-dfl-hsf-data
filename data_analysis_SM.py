# %%
# IMPORTING PACKAGES
import numpy as np

from plotting_common import (
    COLUMNWIDTH_PT,
    TEXTWIDTH_PT,
    configure_matplotlib,
    lighten_color,
    load_npz_dataset,
    moving_time_integral_centered,
    save_figure,
    set_size,
)

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit


def log_fit_func(t, a, b):
    return a * np.log(t) + b


def fit_func_lin(x, a, b):
    return a * x + b


fontsize = 10
configure_matplotlib(fontsize=fontsize)

textwidth_pt = TEXTWIDTH_PT
columnwidth_pt = COLUMNWIDTH_PT

# %%
# SPECTRAL PROPERTIES OF FRAGMENTATION
# fig.S2 fragmentation_spectrum
res = load_npz_dataset("frag_spectrum")
target_height_in = set_size(columnwidth_pt, subplots=(2, 1), height_factor=1.35)[1]
# Define the desired ticks and their labels
custom_xticks = np.arange(-75, 110, 25)
# custom_xticklabels = [r"$-50$", r"$0$", r"$50$", r"$100$"]
# Define the desired ticks and their labels
custom_yticks1 = [1e-1, 1e-5, 1e-9, 1e-13, 1e-17]
custom_yticklabels1 = [
    r"$10^{-1}$",
    r"$10^{-5}$",
    r"$10^{-9}$",
    r"$10^{-13}$",
    r"$10^{-17}$",
]
custom_yticks2 = [0, 2, 4, 6]
custom_yticklabels2 = [r"$0.0$", r"$2.0$", r"$4.0$", r"$6.0$"]

fig, ax = plt.subplots(
    2,
    1,
    figsize=(set_size(columnwidth_pt, subplots=(2, 2))),
    constrained_layout=True,
    sharex="col",
)
ax[0].plot(
    res["energy"],
    res["overlap"],
    "o",
    markersize=0.7,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.4,
)
ax[0].set(ylabel=r"overlap $\mathcal{O}$", yscale="log", ylim=[1e-17, 1])
ax[1].plot(
    res["energy"],
    res["entropy"],
    "o",
    markersize=0.7,
    markeredgecolor="darkblue",
    markerfacecolor="white",
    markeredgewidth=0.4,
)
ax[1].set(ylabel=r"entropy $\mathcal{S}$", xlabel=r"energy $E$")
ax[1].set_xticks(custom_xticks)
# ax[1].set_xticklabels(custom_xticklabels)
ax[0].set_yticks(custom_yticks1)
ax[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
ax[0].set_yticklabels(custom_yticklabels1)
ax[1].set_yticks(custom_yticks2)
ax[1].set_yticklabels(custom_yticklabels2)
ax[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
save_figure("figS2_fragmentation_spectrum")
# %%
# DYNAMICAL PROPERTIES OF FRAGMENTATION
# fig.S3 fragmentation_dynamics
# --- Figure 1: main 2x1 ---
res_dyn = load_npz_dataset("frag_dynamics")
custom_xticks = np.arange(0, 501, 100)
custom_yticks1 = np.arange(0, 1.1, 0.1)
# Make the top axis taller than the bottom (optional, per your request)
fig, ax = plt.subplots(
    2,
    1,
    figsize=set_size(columnwidth_pt, subplots=(2, 1)),
    constrained_layout=True,
    sharex="col",
    gridspec_kw={"height_ratios": [1.0, 1.0]},  # <-- tweak as you like
)
# Top: fidelity
ax[0].plot(
    res_dyn["time_steps"][:1000], res_dyn["overlap"][:1000], "-", lw=0.5, c="darkblue"
)
ax[0].set(ylabel=r"fidelity $\mathcal{F}$")
ax[0].set_yticks(custom_yticks1)
ax[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
# Bottom: entropy
ax[1].plot(
    res_dyn["time_steps"][:1000], res_dyn["entropy"][:1000], "-", lw=0.5, c="darkblue"
)
ax[1].set(
    ylabel=r"entropy $\mathcal{S}$",
    xlabel=r"time $t$",
    yticks=np.arange(0.0, 4.1, 0.5),
    xticks=custom_xticks,
)
ax[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
# --- Inset: QMBS vs HSF inside ax[0] ---
res_in = load_npz_dataset("qmbs_vs_hsf_l8")
# Inset placement (axes-fraction). Tune bbox_to_anchor to move/resize.
axins1 = inset_axes(
    ax[0],
    width="45%",
    height="45%",  # size of inset relative to ax[0]
    loc="lower left",
    bbox_to_anchor=(0.47, 0.42, 1.0, 1.0),
    bbox_transform=ax[0].transAxes,
    borderpad=0.9,
)

# Keep style consistent
axins1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
axins1.set(
    xlim=[-0.2, 14],
    ylim=[0.0, 1.0],
    yticks=np.arange(0.0, 1.1, 0.2),  # fewer ticks looks cleaner in an inset
    xticks=np.arange(0, 15, 2),
)

linestyles = ["--", "-"]
colors = ["red", "darkblue"]
for ii, case in enumerate(["QMBS", "HSF"]):
    axins1.plot(
        res_in["time"],
        res_in[f"F_{case}"],
        linestyles[ii],
        label=rf"{case}",
        linewidth=0.6,
        color=colors[ii],  # inset is usually clearer in monochrome
    )

# Optional: smaller tick labels in inset
axins1.tick_params(labelsize=0.85 * plt.rcParams["xtick.labelsize"])
# Inset legend (better than fig.legend here)
axins1.legend(
    loc="lower left",
    bbox_to_anchor=(-1.0, 0.35),
    ncol=1,
    borderpad=0.2,
    framealpha=0.2,
    facecolor="white",
    edgecolor="black",
    fontsize=0.9 * plt.rcParams["legend.fontsize"],
)

# Inset placement (axes-fraction). Tune bbox_to_anchor to move/resize.
axins2 = inset_axes(
    ax[1],
    width="45%",
    height="45%",  # size of inset relative to ax[0]
    loc="lower left",
    bbox_to_anchor=(0.47, 0.09, 1.0, 1.0),
    bbox_transform=ax[1].transAxes,
    borderpad=0.9,
)

# Keep style consistent
axins2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
axins2.set(
    xlim=[-0.2, 14],
    yticks=np.arange(0.0, 4.1, 1.0),
    xticks=np.arange(0, 15, 2),
)

for ii, case in enumerate(["QMBS", "HSF"]):
    axins2.plot(
        res_in["time"],
        res_in[f"EE_{case}"],
        linestyles[ii],
        label=rf"{case}",
        linewidth=0.6,
        color=colors[ii],  # inset is usually clearer in monochrome
    )

# Optional: smaller tick labels in inset
axins2.tick_params(labelsize=0.85 * plt.rcParams["xtick.labelsize"])
save_figure("figS3_fragmentation_dynamics")
# %%
# ABSENCE OF THERMALIZATION
# fig.S4 populations
res = load_npz_dataset("frag_dynamics")
fig, ax = plt.subplots(
    2,
    1,
    figsize=set_size(columnwidth_pt, subplots=(2, 1), height_factor=0.57),
    sharex=True,
    sharey=False,
    gridspec_kw={"wspace": 0.17},
    constrained_layout=True,
)

ax[1].set(xlabel=r"$t$")
for ii in range(2):
    ax[ii].grid(True, which="both", linestyle="--", linewidth=0.5)
ax[0].set(ylabel=r"population $p_{1}$", yticks=np.arange(0,0.41,0.1), ylim=[-0.01,0.41])
ax[0].plot(
    res["time_steps"],
    res["N_single"],
    "-",
    c="darkblue",
    markeredgewidth=0.1,
    linewidth=0.6,
)
handleDyn = mlines.Line2D(
    [],
    [],
    color="darkblue",
    label=r"$p(t)$",  # or "dynamics", or r"$\langle p_1(t)\rangle$"
    lw=1.5,
)

lineDE = ax[0].axhline(y=res["DE_N_single"], linestyle="-", lw=1, color="darkred")
lineME = ax[0].axhline(y=res["ME_N_single"], linestyle="--", lw=1, color="darkgreen")
lineDE.set_path_effects(
    [
        pe.Stroke(linewidth=2.5, foreground=lighten_color("darkred", 0.25)),
        pe.Normal(),
    ]
)
lineME.set_path_effects(
    [
        pe.Stroke(linewidth=3, foreground=lighten_color("darkgreen", 0.45)),
        pe.Normal(),
    ]
)
ax[1].set(ylabel=r"population $p_{2}$")
ax[1].plot(
    res["time_steps"],
    res["N_pair"],
    "-",
    c="darkblue",
    markeredgewidth=0.1,
    linewidth=1,
)
lineDE = ax[1].axhline(y=res["DE_N_pair"], linestyle="-", lw=1, color="darkred")
lineME = ax[1].axhline(y=res["ME_N_pair"], linestyle="--", lw=1, color="darkgreen")
lineDE.set_path_effects(
    [
        pe.Stroke(linewidth=2.5, foreground=lighten_color("darkred", 0.25)),
        pe.Normal(),
    ]
)
lineME.set_path_effects(
    [
        pe.Stroke(linewidth=3, foreground=lighten_color("darkgreen", 0.45)),
        pe.Normal(),
    ]
)
handleDE = mlines.Line2D(
    [],
    [],
    color="darkred",
    label="DE",
    lw=1,
    path_effects=[
        pe.Stroke(linewidth=5, foreground=lighten_color("darkred", 0.25)),
        pe.Normal(),
    ],
)
handleME = mlines.Line2D(
    [],
    [],
    color="darkgreen",
    label="ME",
    lw=1,
    linestyle="--",
    path_effects=[
        pe.Stroke(linewidth=2, foreground=lighten_color("darkgreen", 0.45)),
        pe.Normal(),
    ],
)
ax[0].legend(
    handles=[handleDyn, handleDE, handleME],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.9),  # Adjust vertical placement (above ax[1])
    ncol=3,  # Single row with 2 items
    frameon=True,
    fontsize=fontsize,
    handlelength=2.5,
    borderpad=0.35,
    framealpha=0.5,
    facecolor="white",
    edgecolor="black",
)
save_figure("figS4_populations")
# %%
# EFFECTIVE MODEL
# fig.S1 effective_model
files = ["g1m1", "g10m1", "g20m1"]
cases = [r"$m=1,\, g^{2}=1$", r"$m=1,\,g^{2}=10$", r"$m=1,\,g^{2}=20$"]
effective_model_data = load_npz_dataset("effective_model_comparison")

fig, ax = plt.subplots(
    1,
    3,
    figsize=(set_size(textwidth_pt, subplots=(1, 3))),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0.08},  # reduce horizontal space between subplots
    constrained_layout=False,  # disable constrained_layout for manual control
)
ax[0].set(xticks=np.arange(0, 101, 20))
for ii, title in enumerate(cases):
    ax[ii].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    su2_series = effective_model_data[f"{files[ii]}_su2"]
    eff_series = effective_model_data[f"{files[ii]}_eff"]
    tline, data = su2_series.T
    eff_tline, eff_data = eff_series.T
    # plot
    ax[ii].plot(eff_tline, eff_data, "-", c="purple", linewidth=0.8, label=r"effective")
    ax[ii].plot(tline, data, "-", c="darkgreen", linewidth=0.8, label=r"SU(2)")
    if ii == 0:
        ax[ii].set(ylabel=r"imbalance $\mathcal{I}(t)$")
    ax[ii].set(xlabel=r"time $t$", xlim=[-1, 101])
    ax[ii].text(
        0.15,
        1.05,
        title,
        transform=ax[ii].transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.2",
            linewidth=0.8,
        ),
    )
fig.legend(
    [r"effective",r"SU(2)"],
    bbox_to_anchor=(0.91, 0.88),
    ncol=1,
    frameon=False,
    labelspacing=0.1,
    bbox_transform=fig.transFigure,
)
save_figure("figS1_effective_model")
# %%
# ENTROPY COMPARISON
res = {}
res["NOBG"] = load_npz_dataset("entropy_nobg")
res["BG"] = load_npz_dataset("entropy_bg")
gscale = np.linspace(0, 15, 20)
sm = cm.ScalarMappable(cmap="magma")
palette = sm.to_rgba(gscale)
def add_fit_box(ax, xy_axes, text, *, fontsize=10, edgecolor="black"):
    ax.text(
        xy_axes[0],
        xy_axes[1],
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox=dict(
            facecolor="white",
            edgecolor=edgecolor,
            boxstyle="round,pad=0.25",
            linewidth=0.8,
            alpha=0.9,
        ),
    )
# ENTANGLEMENT ENTROPY HORIZONTAL
target_height_in = set_size(columnwidth_pt, subplots=(1, 2), height_factor=1.4)[1]
fig, ax = plt.subplots(
    1,
    2,
    figsize=(set_size(textwidth_pt)[0], target_height_in * 1.5),
    sharex=True,
    sharey=False,
    gridspec_kw={"wspace": 0.12},  # reduce horizontal space between subplots
    constrained_layout=False,  # disable constrained_layout for manual control
)
plt.subplots_adjust(left=-0.1, right=0.99, top=0.99, bottom=0.01)
# -------------------------------------------------------------------------------------
# AXES SETTINGS
colorSS = "darkred"
colorGI = "darkblue"
text_colors = [colorGI, colorSS]
title_text = [r"GI", r"SS"]
label_text = [r"(a)", r"(b)"]
smax = [7.34, 11.83]
smax_coords = [(0.05, 0.9), (0.05, 0.96)]
ylim_vals = [9, 13.5]
fit_labels = [
    r"$\mathcal{S}_{\mathrm{GI}}\!=\!0.386(6)t+0.43(2)$",
    r"$\mathcal{S}_{\mathrm{SS}}\!=\!2.054(6)\log(t)+2.63(1)$",
]
ax[0].set(ylabel=r"entropy $\overline{\mathcal{S}}$")
for ii in range(2):
    ax[ii].set(xlabel=r"time $t$", xlim=[-0.2, 10], xticks=np.arange(11))
    ax[ii].set(yticks=np.arange(0, ylim_vals[ii], 2), ylim=[0, ylim_vals[ii]])
    ax[ii].axhline(y=smax[ii], color="black", linestyle="--", linewidth=1.5, zorder=10)
    ax[ii].grid(True, which="both", linestyle="--", linewidth=0.5)
    ax[ii].text(
        0.82,
        1.02,
        title_text[ii],
        transform=ax[ii].transAxes,
        fontsize=fontsize + 2,
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            edgecolor=text_colors[ii],
            boxstyle="round,pad=0.2",
            linewidth=0.8,
        ),
    )
    ax[ii].text(
        0.92,
        0.1,
        label_text[ii],
        transform=ax[ii].transAxes,
        fontsize=fontsize + 2,
        verticalalignment="top",
    )
    ax[ii].text(
        smax_coords[ii][0],
        smax_coords[ii][1],
        r"$\mathcal{S}_{\max}$",
        transform=ax[ii].transAxes,
        fontsize=fontsize,
        verticalalignment="top",
    )
add_fit_box(ax[0], (0.34, 0.11), fit_labels[0], fontsize=fontsize, edgecolor=colorGI)
add_fit_box(ax[1], (0.35, 0.25), fit_labels[1], fontsize=fontsize, edgecolor=colorSS)
# -------------------------------------------------------------------------------------
for jj, g in enumerate(gscale):
    ax[0].plot(
        res["NOBG"]["time_line"][1:],
        moving_time_integral_centered(
            res["NOBG"]["time_line"][1:], res["NOBG"]["entropy"][jj, 1:], 250
        ),
        "o-",
        c=palette[jj],
        markersize=0.05,
        markeredgecolor=palette[jj],
        markerfacecolor="black",
        markeredgewidth=0.1,
        linewidth=1,
    )
    ax[1].plot(
        # res["BG"]["time_line"],#res["BG"]["entropy"][jj, :],
        res["BG"]["time_line"][1:],
        moving_time_integral_centered(
            res["BG"]["time_line"][1:], res["BG"]["entropy"][jj, 1:], 250
        ),
        "o-",
        c=palette[jj],
        markersize=0.05,
        markeredgecolor=palette[jj],
        markerfacecolor="black",
        markeredgewidth=0.1,
        linewidth=1,
    )
cb = fig.colorbar(
    sm, ax=ax, aspect=30, location="right", orientation="vertical", pad=0.02
)
cb.set_label(label=r"$g^{2}$", rotation=0, labelpad=-25, x=-0.2, y=-0.03)
# -------------------------------------------------------------------------------------
# Select the portion of the data
xdata = res["NOBG"]["time_line"][140:600]
ydata = res["NOBG"]["entropy"][19, 140:600]
# Perform the fit
# Define the fitting function: linear in log(x)
def fit_func(logx, a, b):
    return a * logx + b
popt, pcov = curve_fit(fit_func, xdata, ydata)
# popt contains [a, b]
a_fit, b_fit = popt
# Extract standard deviations (errors) from the covariance matrix
perr = np.sqrt(np.diag(pcov))
a_err, b_err = perr
print(f"Fit results:")
print(f"a = {a_fit:.5f} ± {a_err:.5f}")
print(f"b = {b_fit:.5f} ± {b_err:.5f}")
# Plot the fitted line
x_fit = np.linspace(xdata.min(), xdata.max(), 30)
y_fit = fit_func(x_fit, *popt)
ax[0].plot(
    x_fit,
    y_fit,
    "o-",
    c="cyan",
    markersize=0.5,
    markeredgecolor="cyan",
    markerfacecolor="black",
    markeredgewidth=0.1,
    linewidth=1,
    label=f"Fit: $a = {a_fit:.2f}$",
)
# --- select BG fit window (example: mirror your indices, adjust as needed) ---
xdata_bg = res["BG"]["time_line"][30:120]
ydata_bg = moving_time_integral_centered(
    res["BG"]["time_line"][1:], res["BG"]["entropy"][19, 1:], 250
)[30:120]
# ydata_bg = res["BG"]["entropy"][19, 30:120]
# keep only positive times for the log
mask = xdata_bg > 0
xdata_bg = xdata_bg[mask]
ydata_bg = ydata_bg[mask]
# initial guess helps stability (optional but recommended)
p0 = (1.0, np.mean(ydata_bg))
popt_bg, pcov_bg = curve_fit(log_fit_func, xdata_bg, ydata_bg, p0=p0)
a_bg, b_bg = popt_bg
aerr_bg, berr_bg = np.sqrt(np.diag(pcov_bg))
print(f"BG log-fit: a = {a_bg:.5f} ± {aerr_bg:.5f}, b = {b_bg:.5f} ± {berr_bg:.5f}")
# plot fitted curve on ax[1]
x_fit_bg = np.linspace(xdata_bg.min(), xdata_bg.max(), 50)
y_fit_bg = log_fit_func(x_fit_bg, *popt_bg)
ax[1].plot(
    x_fit_bg,
    y_fit_bg,
    "o-",
    c="cyan",
    markersize=0.5,
    markeredgecolor="cyan",
    markerfacecolor="black",
    markeredgewidth=0.1,
    linewidth=1,
    label=rf"$a\ln t + b$ (a={a_bg:.2f})",
)
save_figure("figS5_entropy")
# %%
