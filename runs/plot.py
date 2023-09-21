import json
import ast
import numpy as np
from scipy.stats import sem
import matplotlib
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=9)


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

# Get a colormap
cmap_renyi = plt.colormaps.get_cmap("Accent")
cmap = plt.colormaps.get_cmap("viridis")
alphas = np.array([int(a) for a in alpha_data_simple.keys()])
color_idcs = np.linspace(0, 1, len(alphas))
cs = cmap(color_idcs)

#################

plt.close("all")

fig = plt.figure(figsize=(3.40 * 2, 2.10 - 0.1))
gs = fig.add_gridspec(
    nrows=2,
    ncols=5,
    wspace=0.5,
    hspace=0.25,
    width_ratios=[2.5, 2.5, 0.8, 2.5, 1.5],
    height_ratios=[1, 1],
)

ax2 = fig.add_subplot(gs[:, 0])
ax3 = fig.add_subplot(gs[:, 1])
dummy_ax = fig.add_subplot(gs[:, 2])
dummy_ax_ = fig.add_subplot(gs[:, :2])
ax4 = fig.add_subplot(gs[:, 3])
ax1_qsk = fig.add_subplot(gs[0, 4])
ax1_df = fig.add_subplot(gs[1, 4])

dummy_ax.axis("off")
dummy_ax_.axis("off")

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
ax2.set_ylim(1.0e-6, 1.0e-2)
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
        color=cs[i],
        label=r"$\alpha=" + str(alpha) + "$",
    )

ax3.axhline(y=1.0e-3, color="grey", linestyle="--", linewidth=1, label=r"$10^{-3}$")

ax3.set_xlabel(r"$L$")

fig.subplots_adjust(
    left=0.085, bottom=0.2, right=0.98, top=0.95, wspace=0.35, hspace=None
)

cbar_ax = fig.add_axes([0.49, 0.2, 0.02, 0.75])  # Adjust position and size

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

legend_handles = [matplotlib.lines.Line2D([0], [0], color="black", label="FF")]
legend = matplotlib.legend.Legend(
    ax3, legend_handles, labels=["FF"], frameon=False, loc="upper left"
)
legend.set_bbox_to_anchor((0.4, 0.7))
ax3.add_artist(legend)
legend_handles = [
    matplotlib.lines.Line2D([0], [0], color="black", linestyle="-.", label="FF+SD")
]
legend = matplotlib.legend.Legend(
    ax3, legend_handles, labels=["FF+SD"], frameon=False, loc="lower right"
)
legend.set_bbox_to_anchor((1.025, -0.05))
ax3.add_artist(legend)

ax3.text(
    0.1,
    0.93,
    r"(b)",
    color="black",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax3.transAxes,
)
ax2.text(
    0.1,
    0.93,
    r"(a)",
    color="black",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax2.transAxes,
)
ax4.text(
    0.1,
    0.93,
    r"(c)",
    color="black",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax4.transAxes,
)

ax1_qsk.text(
    0.2,
    0.8,
    r"(d1)",
    color="black",
    fontsize=10,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax1_qsk.transAxes,
)
ax1_df.text(
    0.2,
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

ax4.set_yscale("log")
ax4.set_xlim(10, 20)
ax4.set_xticks([10, 15, 20])
# ax4.set_ylim(1.0e-6, 1.0)

for i, (alpha, curve_data) in enumerate(data_disf_bf.items()):
    N_values = [N for N, _, _ in curve_data]
    mean_values = [mean for _, mean, _ in curve_data]
    sem_values = [sem for _, _, sem in curve_data]

    # Combine and sort the data based on N_values
    sorted_curve_data = sorted(curve_data, key=lambda x: x[0])

    N_values = [N for N, _, _ in sorted_curve_data]
    mean_values = [mean for _, mean, _ in sorted_curve_data]
    sem_values = [sem for _, _, sem in sorted_curve_data]

    ax4.errorbar(
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

    ax4.errorbar(
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

    ax4.errorbar(
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

ax4.axhline(y=1.0e-3, color="grey", linestyle="--", linewidth=1, label=r"$10^{-3}$")

ax4.set_yscale("log")
ax4.set_xlim(10, 20)
ax4.set_xlabel("$L$")
ax4.set_ylabel(r"Mean $|\frac{E_{\theta} - E_0}{E_0}|$")

plt.savefig("figure.pdf", format="pdf", bbox_inches="tight")
