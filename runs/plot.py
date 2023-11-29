import json
import ast
import numpy as np
from scipy.stats import sem
import matplotlib
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=9)


def load_dict(file_path):
    try:
        with open(file_path, "r") as json_file:
            loaded_dict = json.load(json_file)
        return loaded_dict
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_alpha_data(state):
    filename = state + ".out"
    with open(filename, "r") as json_file:
        input_dict = json.load(json_file)

    # Convert string keys to actual tuples using ast.literal_eval()
    input_dict_tuples = {
        ast.literal_eval(key): value for key, value in input_dict.items()
    }

    # Create a dictionary to store values for each unique (N, alpha) pair
    data_by_pair = {}

    # Group values by (N, alpha) pairs
    for (N, seed, alpha), value in input_dict_tuples.items():
        data_by_pair.setdefault((N, alpha), []).append(value)

    # Calculate mean and standard error of the mean for each pair
    dict1 = {}
    for pair, values in data_by_pair.items():
        mean = np.mean(values)
        sem_value = sem(values)
        dict1[pair] = {"mean": mean, "sem": sem_value}

    # Group data by alpha
    alpha_data = {}
    for (N, alpha), values in dict1.items():
        if not np.isnan(values["sem"]):  # Exclude entries with NaN SEM
            alpha_data.setdefault(alpha, []).append((N, values["mean"], values["sem"]))

    return alpha_data


def get_renyi_data(filename):
    with open(filename, "r") as json_file:
        input_dict = json.load(json_file)

    # Initialize empty lists for N values and seeds
    N_values = []
    seed_values = []

    # Iterate through keys in the data dictionary to extract N and seed values
    for key in input_dict.keys():
        # Split the key by comma and remove parentheses
        parts = key.strip("()").split(", ")
        N, seed = int(parts[0]), int(parts[1])

        # Append unique N and seed values to the lists
        if N not in N_values:
            N_values.append(N)
        if seed not in seed_values:
            seed_values.append(seed)

    # Sort the lists for consistency
    N_values.sort()
    seed_values.sort()

    # Initialize an empty array x
    x = np.zeros((len(N_values), len(seed_values)))

    # Fill the array x using the data dictionary
    for i, N in enumerate(N_values):
        for j, seed in enumerate(seed_values):
            key = f"({N}, {seed})"
            if key in input_dict:
                x[i, j] = input_dict[key]

    return x


renyi_data_sk = get_renyi_data("sk_entropy.out")
renyi_data_df = get_renyi_data("df_entropy.out")

alpha_data_sk = get_alpha_data("sk")
alpha_data_simple = get_alpha_data("disf_simple")
alpha_data_bf = get_alpha_data("disf_bf")

data_disf_bf = get_alpha_data("disf_bf_energy_errors")
data_disf_simple = get_alpha_data("disf_simple_energy_errors")
data_sk = get_alpha_data("sk_energy_errors")

Ns_bf = np.array([10, 12, 14, 16, 18, 20])
seeds_bf = np.arange(10)
alphas_bf = [1 / 8, 1 / 4, 1 / 2, 1, 2]
rtols_bf = [5e-4, 1e-3, 5e-3]
np_alphas_bf = np.array(alphas_bf)
n_parameters_bf = (
    0.5 * np.floor(np_alphas_bf[None, :] * Ns_bf[:, None]) * Ns_bf[:, None] ** 2
    + Ns_bf[:, None] ** 2
    + np.floor(np_alphas_bf[None, :] * Ns_bf[:, None]) * Ns_bf[:, None]
)

dict_bf = load_dict("disf_bf_energy_errors.out")
rel_bf = np.zeros((len(Ns_bf), len(seeds_bf), len(alphas_bf)))

for i, N in enumerate(Ns_bf):
    for j, seed in enumerate(seeds_bf):
        for k, alpha in enumerate(alphas_bf):
            rel_bf[i, j, k] = dict_bf[f"({N}, {seed}, {alpha})"]

scaling_bf = np.array(
    [
        [
            [
                np.interp(
                    rtol_bf, rel_bf[i, j, :], n_parameters_bf[i, :], period=np.inf
                )
                for i in range(len(Ns_bf))
            ]
            for j in range(len(seeds_bf))
        ]
        for rtol_bf in rtols_bf
    ]
)


Ns_simp = np.array([10, 12, 14, 16, 18])
seeds_simp = np.arange(10)
alphas_simp = [1, 2, 4, 8, 16, 32, 64]
rtols_simp = [5e-3, 1e-2]
np_alphas_simp = np.array(alphas_simp)
n_parameters_simp = (
    2 * np.floor(np_alphas_simp[None, :] * Ns_simp[:, None])
    + np.floor(np_alphas_simp[None, :] * Ns_simp[:, None]) * Ns_simp[:, None]
)

dict_simp = load_dict("disf_simple_energy_errors.out")
rel_simp = np.zeros((len(Ns_simp), len(seeds_simp), len(alphas_simp)))

for i, N in enumerate(Ns_simp):
    for j, seed in enumerate(seeds_simp):
        for k, alpha in enumerate(alphas_simp):
            rel_simp[i, j, k] = dict_simp[f"({N}, {seed}, {alpha})"]

from scipy.interpolate import InterpolatedUnivariateSpline

scaling_simp_1 = np.array(
    [
        [
            np.interp(
                rtols_simp[0], rel_simp[i, j, :], n_parameters_simp[i, :], period=np.inf
            )
            for i in range(len(Ns_simp[:-1]))
        ]
        for j in range(len(seeds_simp))
    ]
)
xs = np.log(n_parameters_simp[-1, :])
ys = np.log(rel_simp[-1, :, :])
sps = [InterpolatedUnivariateSpline(xs, _ys, k=1, ext=0) for _ys in ys]
col = np.array(
    [
        np.interp(
            rtols_simp[0],
            np.exp(sp(np.log(np.arange(1000, 1_000_000, 10000)))),
            np.arange(1000, 1_000_000, 10000),
            period=np.inf,
        )
        for sp in sps
    ]
).reshape(-1, 1)
scaling_simp_1 = np.hstack((scaling_simp_1, col))

scaling_simp_2 = np.array(
    [
        [
            np.interp(
                rtols_simp[1], rel_simp[i, j, :], n_parameters_simp[i, :], period=np.inf
            )
            for i in range(len(Ns_simp[:-2]))
        ]
        for j in range(len(seeds_simp))
    ]
)
xs = np.log(n_parameters_simp[-2, :])
ys = np.log(rel_simp[-2, :, :])
sps = [InterpolatedUnivariateSpline(xs, _ys, k=1, ext=0) for _ys in ys]
col_1 = np.array(
    [
        np.interp(
            rtols_simp[1],
            np.exp(sp(np.log(np.arange(1000, 1_000_000, 10000)))),
            np.arange(1000, 1_000_000, 10000),
            period=np.inf,
        )
        for sp in sps
    ]
).reshape(-1, 1)

xs = np.log(n_parameters_simp[-1, :])
ys = np.log(rel_simp[-1, :, :])
sps = [InterpolatedUnivariateSpline(xs, _ys, k=1, ext=0) for _ys in ys]
col_2 = np.array(
    [
        np.interp(
            rtols_simp[1],
            np.exp(sp(np.log(np.arange(1000, 1_000_000, 10000)))),
            np.arange(1000, 1_000_000, 10000),
            period=np.inf,
        )
        for sp in sps
    ]
).reshape(-1, 1)

scaling_simp_2 = np.hstack((scaling_simp_2, col_1, col_2))

scaling_simp = np.stack((scaling_simp_1, scaling_simp_2), axis=0)

# Get a colormap
cmap_renyi = plt.colormaps.get_cmap("Accent")
cmap = plt.colormaps.get_cmap("viridis")
alphas = np.array([int(a) for a in alpha_data_simple.keys()])
color_idcs = np.linspace(0, 1, len(alphas) - 1)
cs = cmap(color_idcs)

#################

plt.close("all")

fig = plt.figure(figsize=(3.40 * 2, 2.10 - 0.1))
gs = fig.add_gridspec(
    nrows=3,
    ncols=7,
    wspace=0.35,
    hspace=0.25,
    width_ratios=[2.5, 0.45, 2.5, 1.15, 2.5, 0.0, 1.5],
    height_ratios=[0.1, 1, 1],
)

dummy_ax = fig.add_subplot(gs[0, :])
dummy_ax.axis("off")
ax2 = fig.add_subplot(gs[1:, 0])
dummy_ax = fig.add_subplot(gs[1:, 1])
dummy_ax_ = fig.add_subplot(gs[1:, :1])
dummy_ax.axis("off")
dummy_ax_.axis("off")
ax3 = fig.add_subplot(gs[1:, 2])
dummy_ax = fig.add_subplot(gs[1:, 3])
dummy_ax_ = fig.add_subplot(gs[1:, :3])
dummy_ax.axis("off")
dummy_ax_.axis("off")
ax4 = fig.add_subplot(gs[1:, 4])
dummy_ax = fig.add_subplot(gs[1:, 5])
dummy_ax_ = fig.add_subplot(gs[1:, :5])
dummy_ax.axis("off")
dummy_ax_.axis("off")
ax1_qsk = fig.add_subplot(gs[1, 6])
ax1_df = fig.add_subplot(gs[2, 6])

ax1_qsk.set_ylabel(r"$S_2$")
ax1_df.set_ylabel(r"$S_2$")

ax1_qsk.get_yaxis().set_label_coords(-0.42, 0.5)
ax1_df.get_yaxis().set_label_coords(-0.42, 0.5)

ax1_df.set_xlabel(r"$L$")
ax1_qsk.set_xlim(10, 20)
ax1_df.set_xlim(10, 20)
mean_values = np.mean(renyi_data_sk, axis=1)
sem_values = sem(renyi_data_sk, axis=-1)
ax1_qsk.errorbar(
    np.array([10, 12, 14, 16, 18, 20]),
    mean_values,
    yerr=sem_values,
    fmt="d-",
    markeredgecolor="black",
    markeredgewidth=0.4,
    linewidth=1,
    markersize=3,
    capsize=3,
    color="C3",
    label="QSK",
)
mean_values = np.mean(renyi_data_df, axis=1)
sem_values = sem(renyi_data_df, axis=-1)
ax1_df.errorbar(
    np.array([10, 12, 14, 16, 18, 20]),
    mean_values,
    yerr=sem_values,
    fmt="o-",
    markeredgecolor="black",
    markeredgewidth=0.4,
    linewidth=1,
    markersize=3,
    capsize=3,
    color="C3",
    label="DF",
)
ax1_df.set_xticks(np.array([10, 15, 20]))
ax1_qsk.set_xticks(np.array([10, 15, 20]))
ax1_df.text(
    0.75,
    0.225,
    r"DF",
    color="C3",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax1_df.transAxes,
)
ax1_qsk.text(
    0.75,
    0.225,
    r"QSK",
    color="C3",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax1_qsk.transAxes,
)

ax1_qsk.set_yticks([0.2, 0.4, 0.6])
ax1_df.set_yticks([2, 3])
ax1_df.set_yticklabels(["2.0", "3.0"])


plt.setp(ax1_qsk.get_xticklabels(), visible=False)

ax2.set_yscale("log")
ax2.set_xlim(10, 20)
ax2.set_xticks([10, 15, 20])
ax2.set_ylim(1.0e-6, 1.0)
ax2.set_yticks([1e-6, 1e-4, 1e-2, 1])
for i, (alpha, curve_data) in enumerate(alpha_data_sk.items()):
    N_values = [N for N, _, _ in curve_data]
    mean_values = [mean for _, mean, _ in curve_data]
    sem_values = [sem for _, _, sem in curve_data]

    # Combine and sort the data based on N_values
    sorted_curve_data = sorted(curve_data, key=lambda x: x[0])

    N_values = [N for N, _, _ in sorted_curve_data]
    mean_values = [mean for _, mean, _ in sorted_curve_data]
    sem_values = [sem for _, _, sem in sorted_curve_data]

    ax2.errorbar(
        N_values,
        mean_values,
        yerr=sem_values,
        fmt="d-",
        linewidth=1,
        markersize=3,
        markeredgecolor="black",
        markeredgewidth=0.4,
        capsize=3,
        color=cs[i],
        label=r"$\alpha=" + str(alpha) + "$",
    )


ax3.set_xlim(10, 20)
ax3.set_xticks(np.array([10, 15, 20]))
ax3.set_ylim(1.0e-6, 1.0)
ax3.set_yscale("log")
ax3.set_yticks([1e-6, 1e-4, 1e-2, 1.0])

# Create a plot for each alpha
for i, (alpha, curve_data) in enumerate(alpha_data_sk.items()):
    N_values = [N for N, _, _ in curve_data]
    mean_values = [mean for _, mean, _ in curve_data]
    sem_values = [sem for _, _, sem in curve_data]

    # Combine and sort the data based on N_values
    sorted_curve_data = sorted(curve_data, key=lambda x: x[0])

    N_values = [N for N, _, _ in sorted_curve_data]
    mean_values = [mean for _, mean, _ in sorted_curve_data]
    sem_values = [sem for _, _, sem in sorted_curve_data]

    ax2.errorbar(
        N_values,
        mean_values,
        yerr=sem_values,
        fmt="d-",
        linewidth=1,
        markersize=3,
        markeredgecolor="black",
        markeredgewidth=0.4,
        capsize=3,
        color=cs[i],
        label=r"$\alpha=" + str(alpha) + "$",
    )

# Create a plot for each alpha
for i, (alpha, curve_data) in enumerate(alpha_data_simple.items()):
    N_values = [N for N, _, _ in curve_data]
    mean_values = [mean for _, mean, _ in curve_data]
    sem_values = [sem for _, _, sem in curve_data]

    # Combine and sort the data based on N_values
    sorted_curve_data = sorted(curve_data, key=lambda x: x[0])

    N_values = [N for N, _, _ in sorted_curve_data]
    mean_values = [mean for _, mean, _ in sorted_curve_data]
    sem_values = [sem for _, _, sem in sorted_curve_data]

    if alpha < 32:
        ax2.errorbar(
            N_values,
            mean_values,
            yerr=sem_values,
            fmt="o-",
            linewidth=1,
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.4,
            capsize=3,
            color=cs[i],
            label=r"$\alpha=" + str(alpha) + "$",
        )

ax2.axhline(y=1.0e-3, color="grey", linestyle="--", linewidth=1, label=r"$10^{-3}$")

ax2.set_xlabel(r"$L$")
ax2.set_ylabel("Mean infidelity")

for i, (alpha, curve_data) in enumerate(alpha_data_bf.items()):
    N_values = [N for N, _, _ in curve_data]
    mean_values = [mean for _, mean, _ in curve_data]
    sem_values = [sem for _, _, sem in curve_data]

    # Combine and sort the data based on N_values
    sorted_curve_data = sorted(curve_data, key=lambda x: x[0])

    N_values = [N for N, _, _ in sorted_curve_data]
    mean_values = [mean for _, mean, _ in sorted_curve_data]
    sem_values = [sem for _, _, sem in sorted_curve_data]

    ax2.errorbar(
        N_values,
        mean_values,
        yerr=sem_values,
        fmt="o-.",
        linewidth=1,
        markersize=3,
        markeredgecolor="black",
        markeredgewidth=0.4,
        capsize=3,
        color=cs[i],
        label=r"$\alpha=" + str(alpha) + "$",
    )


fig.subplots_adjust(
    left=0.085, bottom=0.2 - 0.1, right=0.98, top=0.85, wspace=0, hspace=None
)

cbar_ax = fig.add_axes([0.49 + 0.03, 0.1, 0.015, 0.675])  # Adjust position and size

cbar = plt.matplotlib.colorbar.ColorbarBase(
    cbar_ax,
    cmap=cmap,
    orientation="vertical",
    norm=plt.matplotlib.colors.Normalize(0, 4),  # vmax and vmin
    ticks=np.arange(0, 5),
)

cbar.ax.tick_params(axis="y", direction="in")

# Adjust other colorbar properties as needed
cbar.set_label(r"$\alpha$", rotation=0)
cbar.ax.set_yticklabels([1, 2, 4, 8, 16])  # Customize tick labels

legend_handles = [
    matplotlib.lines.Line2D(
        [0], [0], marker="d", linestyle="", color="black", label="QSK"
    ),
    matplotlib.lines.Line2D(
        [0], [0], marker="o", linestyle="", color="black", label="DF"
    ),
    matplotlib.lines.Line2D([0], [0], color="black", label="FF"),
    matplotlib.lines.Line2D([0], [0], color="black", linestyle="-.", label="FF+SD"),
]
legend = matplotlib.legend.Legend(
    ax2,
    legend_handles,
    ncols=4,
    columnspacing=1.2,
    labels=["QSK", "DF", "FF", "FF+SD"],
    frameon=False,
    loc="upper left",
)
legend.set_bbox_to_anchor((-0.15, 1.25))
ax2.add_artist(legend)


ax3.text(
    0.125,
    0.93,
    r"(b)",
    color="black",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax3.transAxes,
)
ax2.text(
    0.125,
    0.93,
    r"(a)",
    color="black",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax2.transAxes,
)
ax4.text(
    0.125,
    0.93,
    r"(c)",
    color="black",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax4.transAxes,
)

ax1_qsk.text(
    0.25,
    0.8,
    r"(d1)",
    color="black",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax1_qsk.transAxes,
)
ax1_df.text(
    0.25,
    0.8,
    r"(d2)",
    color="black",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax1_df.transAxes,
)

data_disf_bf = get_alpha_data("disf_bf_energy_errors")
data_disf_simple = get_alpha_data("disf_simple_energy_errors")
data_sk = get_alpha_data("sk_energy_errors")


cmap = plt.colormaps.get_cmap("viridis")
alphas = np.array([int(a) for a in data_disf_bf.keys()])
color_idcs = np.linspace(0, 1, np.sum(alphas < 32))
cs = cmap(color_idcs)

ax3.set_yscale("log")
ax3.set_xlim(10, 20)
ax3.set_xticks([10, 15, 20])

for i, (alpha, curve_data) in enumerate(data_disf_bf.items()):
    N_values = [N for N, _, _ in curve_data]
    mean_values = [mean for _, mean, _ in curve_data]
    sem_values = [sem for _, _, sem in curve_data]

    # Combine and sort the data based on N_values
    sorted_curve_data = sorted(curve_data, key=lambda x: x[0])

    N_values = [N for N, _, _ in sorted_curve_data]
    mean_values = [mean for _, mean, _ in sorted_curve_data]
    sem_values = [sem for _, _, sem in sorted_curve_data]

    if alpha == 1:
        ax3.errorbar(
            N_values,
            mean_values,
            yerr=sem_values,
            fmt="o-.",
            linewidth=1,
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.4,
            capsize=3,
            color=cs[0],
            label=r"$\alpha=" + str(alpha) + "$",
        )

for i, (alpha, curve_data) in enumerate(data_disf_simple.items()):
    N_values = [N for N, _, _ in curve_data]
    mean_values = [mean for _, mean, _ in curve_data]
    sem_values = [sem for _, _, sem in curve_data]

    # Combine and sort the data based on N_values
    sorted_curve_data = sorted(curve_data, key=lambda x: x[0])

    N_values = [N for N, _, _ in sorted_curve_data]
    mean_values = [mean for _, mean, _ in sorted_curve_data]
    sem_values = [sem for _, _, sem in sorted_curve_data]

    if alpha == 1:
        ax3.errorbar(
            N_values,
            mean_values,
            yerr=sem_values,
            fmt="o-",
            linewidth=1,
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.4,
            capsize=3,
            color=cs[0],
            label=r"$\alpha=" + str(alpha) + "$",
        )

for i, (alpha, curve_data) in enumerate(data_sk.items()):
    N_values = [N for N, _, _ in curve_data]
    mean_values = [mean for _, mean, _ in curve_data]
    sem_values = [sem for _, _, sem in curve_data]

    # Combine and sort the data based on N_values
    sorted_curve_data = sorted(curve_data, key=lambda x: x[0])

    N_values = [N for N, _, _ in sorted_curve_data]
    mean_values = [mean for _, mean, _ in sorted_curve_data]
    sem_values = [sem for _, _, sem in sorted_curve_data]

    if alpha == 1:
        ax3.errorbar(
            N_values,
            mean_values,
            yerr=sem_values,
            fmt="d-",
            linewidth=1,
            markersize=3,
            markeredgecolor="black",
            markeredgewidth=0.4,
            capsize=3,
            color=cs[0],
            label=r"$\alpha=" + str(alpha) + "$",
        )

ax3.axhline(y=1.0e-3, color="grey", linestyle="--", linewidth=1, label=r"$10^{-3}$")

ax3.set_yscale("log")
ax3.set_xlim(10, 20)
ax3.set_xlabel("$L$")
ax3.set_ylabel(r"Mean $|\frac{E_{\theta} - E_0}{E_0}|$")

cmap = plt.colormaps.get_cmap("viridis")
color_idcs = np.linspace(0, 1, 4)
cs = cmap(color_idcs)

for i in range(len(rtols_bf)):
    ax4.plot(
        np.log(Ns_bf),
        np.mean(scaling_bf[i, :, :], axis=-2),
        marker="o",
        linestyle="--",
        color=cs[i],
        linewidth=1,
        markersize=3,
        markeredgecolor="black",
        markeredgewidth=0.4,
    )

for i in range(len(rtols_simp)):
    ax4.plot(
        np.log(Ns_simp),
        np.mean(scaling_simp[i, :, :], axis=-2),
        marker="o",
        color=cs[i + 2],
        linewidth=1,
        markersize=3,
        markeredgecolor="black",
        markeredgewidth=0.4,
    )

ax4.plot(
    np.log(Ns_simp)[-1],
    np.mean(scaling_simp[-1, :, :], axis=-2)[-1],
    marker="o",
    color="none",
    linestyle="",
    markersize=6,
    markeredgecolor="red",
    markeredgewidth=0.4,
)
ax4.plot(
    np.log(Ns_simp)[-2:],
    np.mean(scaling_simp[-2, :, :], axis=-2)[-2:],
    marker="o",
    color="none",
    linestyle="",
    markersize=6,
    markeredgecolor="red",
    markeredgewidth=0.4,
)

ax4.set_ylim(100, 300000)
ax4.set_yscale("log")
ax4.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax4.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax4.set_xticks(np.log(np.array([10, 12, 14, 16, 18, 20])))
ax4.set_xticklabels([r"$\ln 10$", "", r"$\ln 14$", "", "", r"$\ln 20$"])
ax4.set_xlabel(r"$\ln L$")
ax4.set_ylabel("Number of parameters")

cbar_ax = fig.add_axes([0.66 + 0.02, 0.45 - 0.1, 0.01, 0.3])  # Adjust position and size

cbar = plt.matplotlib.colorbar.ColorbarBase(
    cbar_ax,
    cmap=cmap,
    orientation="vertical",
    norm=plt.matplotlib.colors.Normalize(0, 3),  # vmax and vmin
    ticks=np.arange(0, 4),
)

cbar.ax.tick_params(axis="y", direction="in")

cbar.ax.set_yticklabels(
    ["", r"$10^{-3}$", "", r"$10^{-2}$"], fontsize=5
)  # Customize tick labels

ax4.text(
    0.1,
    0.6,
    r"rel. energy tol.",
    color="black",
    fontsize=7,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax4.transAxes,
    rotation=90,
)

plt.savefig("figure.pdf", format="pdf", bbox_inches="tight")
