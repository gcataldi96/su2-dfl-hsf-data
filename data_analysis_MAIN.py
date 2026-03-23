# %%
# IMPORTING PACKAGES
import numpy as np

from plotting_common import (
    COLUMNWIDTH_PT,
    TEXTWIDTH_PT,
    configure_matplotlib,
    lighten_color,
    load_npz_dataset,
    moving_time_integral,
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
# IMBALANCE
# fig.2 imbalance
res = load_npz_dataset("phase_diagram_bg")
# DYNAMICS
res1 = load_npz_dataset("dynamics_dfl")
res2 = load_npz_dataset("frag_imbalance")
fig, ax = plt.subplots(
    1,
    4,
    figsize=set_size(textwidth_pt, subplots=(1, 4), height_factor=1.7),
    constrained_layout=True,
    sharex=False,
    sharey=False,
)
X = np.transpose(res[f"DE_N_tot"])[:22, :22]
img = ax[0].imshow(X, cmap="magma", origin="lower", extent=[0, 10, 0, 10])
ax[0].set(
    ylabel=r"$m$",
    xlabel=r"$g^{2}$",
    xticks=np.arange(0, 11, 1),
    yticks=np.arange(0, 11, 1),
)
# create a cbar axis glued to the TOP of axs[0,0], same inner width
cax = inset_axes(
    ax[0],
    width="100%",
    height="7%",  # thickness via height
    loc="lower left",
    bbox_to_anchor=(
        0.0,
        1.05,
        1.0,
        0.42,
    ),  # (x0,y0,w,h) in axes coords; y0>1 places it above
    bbox_transform=ax[0].transAxes,
    borderpad=0,
)
cb = fig.colorbar(img, cax=cax, orientation="horizontal")
# put ticks on top and label on top
cb.ax.xaxis.set_ticks_position("top")
cb.ax.xaxis.set_label_position("top")
cb.set_label(r"$\mathcal{I}_{DE}$", labelpad=3)
cb.set_ticks(np.arange(0, 0.26, 0.1))
ax[2].sharey(ax[1])
ax[2].tick_params(left=False, labelleft=False)
ax[3].sharey(ax[1])
ax[3].tick_params(left=False, labelleft=False)
# Define colors and line width
colorSS = "red"
colorGI = "darkblue"
horizontal_colorSS = lighten_color(colorSS, 0.15)  # adjust the factor as desired
horizontal_colorGI = lighten_color(colorGI, 0.15)  # adjust the factor as desired
line_width = 0.8  # adjust the line width as desired
panel_labels = ["(b)", "(c)", "(d)"]
ls_DE_SS = "-."  # plateau DE
ls_time_SS = "--"  # curva nel tempo
ls_DE_GI = "-"  # plateau DE
ls_time_GI = "-"  # curva nel tempo
line_width_GI = 1.1  # adjust the line width as desired
line_width_SS = 0.8
for ii, g in enumerate([1, 10]):
    # GI plateau
    line4 = ax[ii + 1].axhline(
        y=res1["DE_N_tot"][1, ii],
        linestyle=ls_DE_GI,
        lw=line_width_GI,
        color=colorGI,
    )
    line4.set_path_effects(
        [
            pe.Stroke(linewidth=3, foreground=lighten_color(colorGI, 0.15)),
            pe.Normal(),
        ]
    )

    # GI curva nel tempo
    ax[ii + 1].plot(
        res1["tline"],
        moving_time_integral(res1["tline"], res1["delta"][1, ii, :], 300),
        linestyle=ls_time_GI,
        lw=line_width_GI,
        color=colorGI,
    )
    # SS plateau (orizzontale)
    line3 = ax[ii + 1].axhline(
        y=res1["DE_N_tot"][0, ii],
        linestyle=ls_DE_SS,
        lw=line_width_SS,
        color=colorSS,
    )
    line3.set_path_effects(
        [
            pe.Stroke(linewidth=5, foreground=lighten_color(colorSS, 0.15)),
            pe.Normal(),
        ]
    )
    # SS curva nel tempo
    ax[ii + 1].plot(
        res1["tline"],
        moving_time_integral(res1["tline"], res1["delta"][0, ii, :], 300),
        linestyle=ls_time_SS,
        lw=line_width_SS,
        color=colorSS,
    )

line4 = ax[2 + 1].axhline(
    y=res2["DE_N_tot"], linestyle=ls_DE_GI, lw=line_width_GI, color=colorGI
)
line4.set_path_effects(
    [
        pe.Stroke(linewidth=3, foreground=horizontal_colorGI),
        pe.Normal(),
    ]
)
ax[2 + 1].plot(
    res2["time_steps"],
    moving_time_integral(res2["time_steps"], res2["delta"], 300),
    linestyle=ls_time_GI,
    lw=line_width_GI,
    color=colorGI,
    label="GI",
)

# Fourth horizontal line:
line4 = ax[2 + 1].axhline(
    y=res2["bg_DE_N_tot"], linestyle=ls_DE_SS, lw=line_width_SS, color=colorSS
)
line4.set_path_effects(
    [
        pe.Stroke(linewidth=5, foreground=horizontal_colorSS),
        pe.Normal(),
    ]
)
ax[2 + 1].plot(
    res2["bg_time_steps"],
    moving_time_integral(res2["bg_time_steps"], res2["bg_delta"], 300),
    linestyle=ls_time_SS,
    lw=line_width_SS,
    color=colorSS,
    label="SS",
)

title_texts = [
    r" $\,m\!=\!1,\, g^{2}\!=\!1\,$ ",
    r" $\,m\!=\!10,\, g^{2}\!=\!10\,$ ",
    r" $\,m\!=\!10,\, g^{2}\!=\!2\,$ ",
]
num_list = [r"$\;1$", r"$2$", r"$3$"]
xcoords_list = [0.1, 0.96, 0.2]
ycoords_list = [0.1, 0.96, 0.96]
for ii in range(1, 4, 1):
    if ii == 1:
        ax[ii].set(ylabel=r"imbalance $\overline{\mathcal{I}}(t)$")
    ax[ii].set(
        xticks=np.arange(0, 1001, 250),
        xlabel=r"time $t$",
        yticks=np.arange(-0.25, 1.1, 0.25),
    )
    ax[ii].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax[ii].text(
        0.1,
        0.91,
        title_texts[ii - 1],
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
    ax[ii].text(
        0.9,
        0.86,
        num_list[ii - 1],
        transform=ax[ii].transAxes,
        fontsize=fontsize,
        ha="center",
        va="center",  # center text in the circle
        bbox=dict(
            boxstyle="circle,pad=0.1",  # ← circular box (tune pad for size)
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
        ),
    )
    ax[0].text(
        xcoords_list[ii - 1],
        ycoords_list[ii - 1],
        num_list[ii - 1],
        transform=ax[0].transAxes,
        fontsize=fontsize,
        ha="center",
        va="center",  # center text in the circle
        bbox=dict(
            boxstyle="circle,pad=0.1",  # ← circular box (tune pad for size)
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
        ),
    )
ax[1].yaxis.set_label_coords(-0.3, 0.5)
# Create custom legend handles for the two main curves.
handle_SS_DE = mlines.Line2D(
    [],
    [],
    color=colorSS,
    linestyle=ls_DE_SS,
    lw=line_width_SS,
    label=r"SS $(\mathrm{DE})$",
    path_effects=[
        pe.Stroke(linewidth=5, foreground=lighten_color(colorSS, 0.25)),
        pe.Normal(),
    ],
)
handle_SS_time = mlines.Line2D(
    [],
    [],
    color=colorSS,
    linestyle=ls_time_SS,
    lw=line_width_SS,
    label=r"SS $(t)$",
)
handle_GI_DE = mlines.Line2D(
    [],
    [],
    color=colorGI,
    linestyle=ls_DE_GI,
    lw=line_width_GI,
    label=r"GI $(\mathrm{DE})$",
    path_effects=[
        pe.Stroke(linewidth=5, foreground=lighten_color(colorGI, 0.25)),
        pe.Normal(),
    ],
)

handle_GI_time = mlines.Line2D(
    [],
    [],
    color=colorGI,
    linestyle=ls_time_GI,
    lw=line_width_GI,
    label=r"GI $(t)$",
)
# Place the legend above the subplots, outside the axes area, without a box.
fig.legend(
    handles=[handle_GI_DE, handle_GI_time, handle_SS_DE, handle_SS_time],
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.65, 1.15),
    frameon=False,
)
save_figure("fig2_imbalance")
# %%
# fig3_populations.pdf
res_pd = load_npz_dataset("phase_diagram_nobg")
res = {}
res["NOBG"] = load_npz_dataset("population_scan_nobg")
res["BG"] = load_npz_dataset("population_scan_bg")

fig, ax = plt.subplots(
    1,
    3,
    figsize=set_size(textwidth_pt, subplots=(1, 3), height_factor=1.4),
    constrained_layout=True,
    sharex=False,
    sharey=False,
)
X = np.transpose(res_pd[f"diff"])[:22, :22]
img = ax[0].imshow(X, cmap="magma", origin="lower", extent=[0, 10, 0, 10])
ax[0].set(
    ylabel=r"$m$",
    xlabel=r"$g^{2}$",
    xticks=np.arange(0, 11, 1),
    yticks=np.arange(0, 11, 1),
)
# create a cbar axis glued to the TOP of axs[0,0], same inner width
cax = inset_axes(
    ax[0],
    width="100%",
    height="7%",  # thickness via height
    loc="lower left",
    bbox_to_anchor=(
        0.0,
        1.05,
        1.0,
        0.42,
    ),  # (x0,y0,w,h) in axes coords; y0>1 places it above
    bbox_transform=ax[0].transAxes,
    borderpad=0,
)
cb = fig.colorbar(img, cax=cax, orientation="horizontal")
# put ticks on top and label on top
cb.ax.xaxis.set_ticks_position("top")
cb.ax.xaxis.set_label_position("top")
cb.set_label(r"$\Delta_{p}$", labelpad=4)
xtickscb = np.arange(0, 0.26, 0.05)
cb.set_ticks(xtickscb)
ax[0].axhline(y=1, color="white", linestyle="--", linewidth=1.5, zorder=10)
# AX 1,2
ax[1].set(xticks=np.arange(0.0, 15.1, 2.5))
ax[2].set(xticks=np.arange(0.0, 15.1, 2.5))
obs_names = [r"$p_{\rm{1}}$", r"$p_{\rm{2}}$"]
sizes = [3, 3]
styles = ["--", "-"]
colors = ["darkgreen", "purple"]
gscale = np.linspace(0, 15, 20)
legend_list = []
for ii, obs in enumerate(["N_single", "N_pair"]):
    ax[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    (line,) = ax[1].plot(
        gscale,
        res["NOBG"][f"ME_{obs}"],
        styles[ii],
        c=colors[ii],
        linewidth=1,
        label=rf"{obs_names[ii]}(ME)",
    )
    legend_list.append(line)
    (line,) = ax[1].plot(
        gscale,
        res["NOBG"][f"DE_{obs}"],
        f"{styles[ii]}o",
        c=colors[ii],
        linewidth=1,
        markeredgecolor=colors[ii],
        markerfacecolor="white",
        markeredgewidth=0.5,
        markersize=sizes[ii],
        label=rf"{obs_names[ii]}(DE)",
    )
    legend_list.append(line)
    ax[1].set(xlabel=r"$g^{2}$", ylabel=r"populations $p_{q}$")
    ax[2].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax[2].set(yticks=np.arange(0.275, 0.42, 0.025))
    ax[2].plot(
        gscale,
        res["BG"][f"ME_{obs}"],
        styles[ii],
        c=colors[ii],
        linewidth=1,
    )
    ax[2].plot(
        gscale,
        res["BG"][f"DE_{obs}"],
        f"{styles[ii]}o",
        c=colors[ii],
        linewidth=1,
        markeredgecolor=colors[ii],
        markerfacecolor="white",
        markeredgewidth=0.5,
        markersize=sizes[ii],
    )
    ax[2].set(xlabel=r"$g^{2}$")
fig.text(
    0.01,
    1.18,
    r"(a)",  # position in axes coordinates (x, y)
    color="black",
    fontsize=12,
    fontweight="bold",
    va="top",
    ha="left",
)
fig.text(
    0.34,
    1.18,
    r"(b)",
    color="black",
    fontsize=12,
    fontweight="bold",
    va="top",
    ha="left",
)
colorSS = "darkred"
colorGI = "darkblue"
text_colors = [colorSS, colorGI]
title_text = [r"GI", r"SS"]
for ii in range(2):
    ax[ii + 1].text(
        0.27,
        1.03,
        title_text[ii],
        transform=ax[ii + 1].transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            edgecolor=text_colors[ii],
            boxstyle="round,pad=0.2",
            linewidth=0.8,
        ),
    )
# Place the legend above the subplots, outside the axes area, without a box.
fig.legend(
    handles=legend_list,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.7, 1.2),
    frameon=False,
)
save_figure("fig3_populations")
# %%
# ENTANGLEMENT ENTROPY
res = {}
res["NOBG"] = load_npz_dataset("entropy_nobg")
res["BG"] = load_npz_dataset("entropy_bg")
gscale = np.linspace(0, 15, 20)
sm = cm.ScalarMappable(cmap="magma")
palette = sm.to_rgba(gscale)
target_height_in = set_size(columnwidth_pt, subplots=(1, 2), height_factor=1.5)[1]
fig, ax = plt.subplots(
    1,
    1,
    figsize=(set_size(columnwidth_pt, subplots=(1, 1))),
    constrained_layout=False,  # disable constrained_layout for manual control
)
# plt.subplots_adjust(left=-0.1, right=0.99, top=0.99, bottom=0.01)
ax.set(
    xlabel=r"$t$",
    ylabel=r"entropy $\mathcal{S}$",
    yticks=np.arange(0.0, 12.1, 1.0),
    xscale="log",
)
for jj, g in enumerate(gscale):
    # Plot the entropy for each g value
    ax.plot(
        res["BG"]["time_line"][1:],
        res["BG"]["entropy"][jj, 1:],
        "-",
        c=palette[jj],
        markersize=0.01,
        markeredgecolor=palette[jj],
        markerfacecolor="black",
        markeredgewidth=0.1,
        linewidth=1,
    )
ax.axhline(y=11.83, color="black", linestyle="--", linewidth=1.5, zorder=10)
colorSS = "darkblue"
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.text(
    0.9,
    0.15,
    r"SS",
    transform=ax.transAxes,
    fontsize=fontsize,
    verticalalignment="top",
    bbox=dict(
        facecolor="white",
        edgecolor=colorSS,
        boxstyle="round,pad=0.2",
        linewidth=0.8,
    ),
)
ax.text(
    0.05,
    0.9,
    r"$\mathcal{S}_{\max}$",
    transform=ax.transAxes,
    fontsize=fontsize,
    verticalalignment="top",
)
cb = fig.colorbar(
    sm, ax=ax, aspect=30, location="right", orientation="vertical", pad=0.03
)
cb.set_label(label=r"$g^{2}$", rotation=0, labelpad=-25, x=-0.3, y=-0.03)
cb.set_ticks(np.arange(0.0, 15.1, 2.5))


from scipy.optimize import curve_fit

# Select the portion of the data
xdata = res["BG"]["time_line"][30:120]
ydata = res["BG"]["entropy"][19, 30:120]


# Define the fitting function: linear in log(x)
def fit_func(logx, a, b):
    return a * logx + b


# Take the log of xdata
logxdata = np.log(xdata)

# Perform the fit
popt, pcov = curve_fit(fit_func, logxdata, ydata)

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
logx_fit = np.log(x_fit)
y_fit = fit_func(logx_fit, *popt)

ax.plot(
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
save_figure("fig4_entropy_SS")
# %%
